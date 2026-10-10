import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / 'multiagent/outputs_mine/interp/anthropic_assessment'
sys.path.insert(0, str(ROOT / 'scripts/interp_security'))
sys.path.insert(0, str(ROOT))
from experiment import PAIRS
from runtime import Runner, prompt_ids, left_pad
from multiagent.dataset.c2c import generation_prompt_ids
from multiagent.model.unaware_comm import tuple_to_dynamic_cache

SUFFIX = '\n\nAnswer with Yes or No.'
BATCH = 16


def answer_ids(tok):
    out = {}
    for label in ('Yes', 'No'):
        ids = set()
        for v in (label, ' ' + label):
            t = tok(v, add_special_tokens=False)['input_ids']
            ids.add(t[0])
        out[label] = sorted(ids)
    assert not set(out['Yes']) & set(out['No'])
    return out


def logodds(logits, ids):
    lp = F.log_softmax(logits.float(), -1)
    yes = torch.logsumexp(lp[:, ids['Yes']], -1); no = torch.logsumexp(lp[:, ids['No']], -1)
    return (yes - no).tolist()


@torch.inference_mode()
def native(model, tok, prompts, device):
    ids, mask = left_pad([prompt_ids(tok, p, True) for p in prompts], tok.pad_token_id, device)
    pos = (mask.cumsum(1) - 1).clamp_min(0)
    return model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]


@torch.inference_mode()
def swat(runner, prompts, header):
    caches = [runner.render(p) for p in prompts]
    lengths = [c[0][0].shape[2] for c in caches]
    width = max(lengths)
    packed = tuple(tuple(torch.cat([F.pad(c[layer][kv], (0, 0, width - n, 0)) for c, n in zip(caches, lengths)], 0)
                         for kv in (0, 1)) for layer in range(len(caches[0])))
    cache = tuple_to_dynamic_cache(packed, runner.model.config)
    dev = runner.device
    mask = torch.zeros((len(prompts), width + len(header)), dtype=torch.long, device=dev)
    for i, n in enumerate(lengths):
        mask[i, width - n:] = 1
    ids = torch.tensor([header] * len(prompts), device=dev)
    pos = torch.tensor(lengths, device=dev)[:, None] + torch.arange(len(header), device=dev)[None]
    return runner.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                        logits_to_keep=1).logits[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True); ap.add_argument('--arms', nargs='+', default=['sender', 'receiver', 'swat'])
    ap.add_argument('--sets', type=int, default=None); ap.add_argument('--out', default=str(CORE))
    a = ap.parse_args()
    items = [json.loads(l) for l in (CORE / 'data/persona_sample.jsonl').read_text().splitlines() if l.strip()]
    sets = sorted({it['set'] for it in items})[:a.sets]
    items = [it for it in items if it['set'] in set(sets)]
    out = Path(a.out) / a.pair; out.mkdir(parents=True, exist_ok=True)
    runner = Runner(a.pair, 'renderer' if 'swat' in a.arms else 'receiver')
    for arm in a.arms:
        target = out / f'{arm}.jsonl'
        done = {(json.loads(l)['set'], json.loads(l)['idx']) for l in target.read_text().splitlines()} if target.exists() else set()
        if arm == 'sender':
            model, tok = runner.sender, runner.sender_tok
        else:
            model, tok = runner.model, runner.receiver_tok
        ids = answer_ids(tok); header = generation_prompt_ids(tok)
        todo = [it for it in items if (it['set'], it['idx']) not in done]
        print(f'{a.pair}/{arm}: {len(done)} done, {len(todo)} to go', flush=True)
        with target.open('a') as f:
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                prompts = [it['question'] + SUFFIX for it in chunk]
                logits = swat(runner, prompts, header) if arm == 'swat' else native(model, tok, prompts, runner.device)
                for it, lo in zip(chunk, logodds(logits, ids)):
                    f.write(json.dumps({'set': it['set'], 'idx': it['idx'], 'matching': it['matching'], 'logodds_yes': lo}) + '\n')
                if (i // BATCH) % 50 == 0:
                    f.flush(); print(f'{a.pair}/{arm}: {i + len(chunk)}/{len(todo)}', flush=True)
        print(f'{a.pair}/{arm}: done', flush=True)


if __name__ == '__main__':
    main()
