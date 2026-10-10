from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.abl_closed_form.mapper import ClosedFormMapper
from multiagent.evaluation import renderer as evaluator
from multiagent.model import cache_renderer


def install_adapter(identical_tokenizers):
    original_factory = cache_renderer.renderer_from_config

    def factory(config):
        if config.get('arch') != 'closed-form-ridge':
            return original_factory(config)
        return ClosedFormMapper(**{k: v for k, v in config.items() if k != 'arch'})

    cache_renderer.renderer_from_config = factory
    original_encode = evaluator.encode_context_pair

    def encode(*args, **kwargs):
        context = original_encode(*args, **kwargs)
        if identical_tokenizers:
            if context['sender_ids'] != context['receiver_ids']:
                raise ValueError('Shared-tokenizer closed-form evaluation requires identical source/target token IDs')
            context['virtual_positions'] = np.arange(len(context['sender_ids']), dtype=np.float32)
        return context

    evaluator.encode_context_pair = encode


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--baseline-reference', type=Path)
    extra, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = evaluator.parse_args()
    if not args.output:
        raise ValueError('--output is required')
    if Path(args.output).exists() and not args.resume_from:
        raise FileExistsError(args.output)
    reference = json.loads(extra.baseline_reference.read_text()) if extra.baseline_reference else None
    if reference:
        ref_config = reference['evaluation_config']
        ref_args = ref_config.get('arguments', ref_config)
        for key in ['tasks', 'seed', 'max_length', 'max_new_tokens', 'limit_per_dataset']:
            if ref_args.get(key) is not None and getattr(args, key) != ref_args[key]:
                raise ValueError(f'Reference mismatch for {key}: {getattr(args, key)} != {ref_args[key]}')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = Path(args.renderer)
    if checkpoint.is_dir():
        checkpoint /= 'renderer.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['renderer_config']['arch'] != 'closed-form-ridge':
        raise ValueError('Expected a closed-form checkpoint')
    metadata = payload['closed_form_metadata']
    identical = bool(metadata.get('alignment', {}).get('identical_tokenizers', True))
    if not identical and not payload.get('args', {}).get('causal_cross_attention'):
        raise ValueError('Cross-tokenizer checkpoint must carry causal_cross_attention so alignment limits are passed')
    del payload
    install_adapter(identical)
    original_save = evaluator.save_json

    def save(path, data):
        data = dict(data)
        data['mode'] = 'closed_form_ridge'
        data['closed_form_metadata'] = metadata
        data['baseline_reference'] = str(extra.baseline_reference) if reference else None
        data['adapter_script'] = str(Path(__file__).resolve())
        if reference:
            expected_ids = {r['id'] for r in reference['rows']}
            ids = [r['id'] for r in data['rows']]
            if len(set(ids)) != len(ids) or not set(ids).issubset(expected_ids):
                raise ValueError('Evaluated sample IDs differ from the reference')
            data['sample_ids_match_reference'] = set(ids) == expected_ids
        original_save(path, data)

    evaluator.save_json = save
    evaluator.main()
    if reference:
        result = json.loads(Path(args.output).read_text())
        if not result['sample_ids_match_reference']:
            raise RuntimeError('Evaluation finished without covering all reference samples')


if __name__ == '__main__':
    main()
