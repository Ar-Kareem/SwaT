import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
from multiagent.dataset.c2c import generation_prompt_ids, turn_end_ids
from multiagent.dataset.c2c_cross_family import (
    tokenize_with_offsets, char_map_anchors, virtual_positions, cross_causal_limits,
)
from multiagent.model.cache_renderer import load_renderer_checkpoint, flatten_sender_cache, pred_to_cache
from multiagent.model.unaware_comm import full_sender_cache, tuple_to_dynamic_cache
from multiagent.model.vanilla_loader import load_causal_model
from scripts.vision.consensus import VOTERS
from scripts.vision.run import GEMMA_BASELINES, GEMMA_CONTROLS

CAPTION_PROMPT = (
    'Describe this image in detail for someone who cannot see it. Include visible objects, '
    'counts, colors, spatial relationships, actions, and any legible text or diagram labels. '
    'Report only what is visible; do not invent details. Keep the description within 150 words.'
)
OCR_PROMPT = ('Read the text in this image. Copy visible words and numbers exactly. '
              'Do not describe the image. Output only copied text.')
CAPTION_BUDGET = 256
INSTRUCTION = 'Answer the question with a short phrase, without explanation.'
SYSTEM_DATE = 'September 15, 2026'
REASONING_PROMPT = ('Think through the question step by step using the image. '
                    'Use this turn for reasoning; you will be asked for a final answer next.')


def gemma_control_recipe(method):
    if method not in GEMMA_CONTROLS:
        raise ValueError(f'Unknown Gemma control: {method}')
    return {'version': 1, 'image_retained': True, 'decoding': 'greedy',
            'preparation': 'question-aware reasoning' if method == 'gemma_cot' else 'saved assistant text replay before Q',
            'saved_text': 'S' if method == 'gemma_s' else ('S+OCR' if method == 'gemma_st' else 'none'),
            **({'reasoning_prompt': REASONING_PROMPT,
                'reasoning_budget': 'saved S output_tokens + saved OCR output_tokens per image bundle',
                'reasoning_stop': 'EOS or budget; close assistant turn and request final answer',
                'reasoning_replay': 'exact generated token IDs', 'final_prompt': INSTRUCTION}
               if method == 'gemma_cot' else {})}


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def on_device(batch, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in batch.items()}

def decode(model, tokenizer, cache, prefix_length, max_new_tokens, device):
    ids = generation_prompt_ids(tokenizer)
    if not ids:
        raise ValueError("Empty assistant header")
    stop = turn_end_ids(tokenizer, model.generation_config)
    tokens = torch.tensor([ids], device=device)
    pos = prefix_length
    generated = []
    reason = "cutoff"
    for _ in range(max_new_tokens):
        output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones((1, pos + tokens.shape[1]), device=device, dtype=torch.long),
                       position_ids=torch.arange(pos, pos + tokens.shape[1], device=device)[None])
        cache = output.past_key_values
        pos += tokens.shape[1]
        next_id = int(output.logits[0, -1].argmax())
        if next_id in stop:
            reason = "eos"
            break
        generated.append(next_id)
        tokens = torch.tensor([[next_id]], device=device)
    return {"output": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "output_ids": generated, "stop_reason": reason}


def pad_gemma_inputs(inputs, pad_token_id):
    width = max(x['input_ids'].shape[1] for x in inputs)
    batch = {name: torch.full((len(inputs), width), pad_token_id if name == 'input_ids' else 0,
                             dtype=inputs[0][name].dtype)
             for name in ('input_ids', 'attention_mask', 'token_type_ids')}
    for i, item in enumerate(inputs):
        length = item['input_ids'].shape[1]
        for name in batch:
            batch[name][i, -length:] = item[name][0]
    batch['pixel_values'] = torch.cat([x['pixel_values'] for x in inputs], dim=0)
    return batch


def decode_batch(model, tokenizer, cache, attention_mask, budgets, device):
    header = generation_prompt_ids(tokenizer)
    if not header or not budgets or any(type(n) is not int or n < 1 for n in budgets):
        raise ValueError('A nonempty assistant header and positive per-row budgets are required')
    stop = turn_end_ids(tokenizer, model.generation_config)
    count = len(budgets)
    lengths = attention_mask.sum(dim=1)
    tokens = torch.tensor([header] * count, device=device)
    generated = [[] for _ in budgets]
    reasons, active = ['cutoff'] * count, [True] * count
    consumed = 0
    for step in range(max(budgets)):
        extension = torch.tensor(active, device=device, dtype=attention_mask.dtype)[:, None]
        attention_mask = torch.cat((attention_mask, extension.expand(-1, tokens.shape[1])), dim=1)
        positions = lengths[:, None] + consumed + torch.arange(tokens.shape[1], device=device)[None]
        output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                       attention_mask=attention_mask, position_ids=positions, logits_to_keep=1)
        cache = output.past_key_values
        consumed += tokens.shape[1]
        next_ids = output.logits[:, -1].argmax(dim=-1).tolist()
        for i, next_id in enumerate(next_ids):
            if not active[i]:
                continue
            if next_id in stop:
                reasons[i], active[i] = 'eos', False
            else:
                generated[i].append(next_id)
                if step + 1 == budgets[i]:
                    active[i] = False
        if not any(active):
            break
        tokens = torch.tensor([[t if live else tokenizer.pad_token_id]
                               for t, live in zip(next_ids, active)], device=device)
    return [{'output': tokenizer.decode(ids, skip_special_tokens=True).strip(),
             'output_ids': ids, 'stop_reason': reason} for ids, reason in zip(generated, reasons)]

def freeze_system_date(tokenizer):
    expression = "strftime_now('%B %d, %Y')"
    template = tokenizer.chat_template
    if expression in template:
        tokenizer.chat_template = template.replace(expression, json.dumps(SYSTEM_DATE))
        return True
    if 'strftime_now' in template:
        raise ValueError('Unsupported chat-template date expression in receiver tokenizer')
    return False

def receiver_ids(tokenizer, question, caption=None):
    text = question + '\n' + INSTRUCTION
    if caption is not None:
        text = 'Image description:\n' + caption + '\n\nQuestion:\n' + text
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                            tokenize=False, add_generation_prompt=False)
    return tokenizer(rendered, add_special_tokens=False, return_tensors='pt')['input_ids']

_BOUNDARY_PROBES = ('What is shown?', 'Is this a car?', '"Quoted" start?', '1 + 1?', 'A. option',
                    '(paren) start', 'Image description:\nA cat.\n\nQuestion:\nWhat?')
_BOUNDARY_CACHE = {}


def prefix_keeps_newline(tok):
    key = (tok.name_or_path, tok.__class__.__name__, len(tok))
    if key not in _BOUNDARY_CACHE:
        caption = 'A red car parked on a street.\nThere are two people.'
        base = _prefix_text(tok, caption)
        for keep, prefix in ((False, base), (True, base + '\n')):
            ids = tok(prefix, add_special_tokens=False)['input_ids']
            if all(receiver_ids(tok, q, caption)[0, :len(ids)].tolist() == ids for q in _BOUNDARY_PROBES):
                _BOUNDARY_CACHE[key] = keep
                break
        else:
            raise ValueError('No question-independent receiver prefix boundary for this tokenizer')
    return _BOUNDARY_CACHE[key]


def _prefix_text(tok, caption):
    content = 'Image description:\n' + caption + '\n\nQuestion:'
    marker = '__QUESTION_SENTINEL__'
    text = tok.apply_chat_template([{'role': 'user', 'content': content + '\n' + marker}], tokenize=False, add_generation_prompt=False)
    return text[:text.index(marker) - 1]


def receiver_prefix(tok, caption):
    text = _prefix_text(tok, caption)
    return text + '\n' if prefix_keeps_newline(tok) else text

def caption_layout(processor, tok, image, caption, prompt=CAPTION_PROMPT):
    assert caption.strip() == caption and caption
    messages = [{'role': 'user', 'content': image_content(image, prompt, numbered=False)}]
    original = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    header = processor.tokenizer.decode(generation_prompt_ids(processor.tokenizer), skip_special_tokens=False)
    text = original + header + caption
    batch = processor(text=text, images=image, add_special_tokens=False, return_tensors='pt',
                      return_offsets_mapping=True, do_pan_and_scan=False)
    offsets = [tuple(x) for x in batch.pop('offset_mapping')[0].tolist()]
    expanded = text.replace(processor.boi_token, processor.full_image_sequence)
    sids, checked = tokenize_with_offsets(processor.tokenizer, expanded)
    assert sids == batch['input_ids'][0].tolist() and checked == offsets
    original_ids = caption_inputs(processor, image, prompt, numbered=False)['input_ids'][0].tolist()
    expected_start = original_ids + generation_prompt_ids(processor.tokenizer)
    assert sids[:len(expected_start)] == expected_start
    start, stop = len(expanded) - len(caption), len(expanded)
    content_mask = np.array([a >= start and b <= stop and b > a for a, b in offsets])
    assert np.flatnonzero(content_mask)[0] == len(expected_start)
    rt = receiver_prefix(tok, caption)
    rlist, roffsets = tokenize_with_offsets(tok, rt)
    rstart = rt.index('Image description:\n') + len('Image description:\n')
    assert rt[rstart:rstart+len(caption)] == caption
    regions = [(start, stop, rstart, rstart+len(caption))]
    xs, ys = char_map_anchors(regions, len(expanded), len(rt))
    mpos = torch.tensor(virtual_positions(offsets, roffsets, xs, ys, 'interp')[None], dtype=torch.float32)
    limits = torch.tensor(cross_causal_limits(offsets, roffsets, xs, ys)[None])
    rids = torch.tensor([rlist])
    count = len(image) if isinstance(image, (list, tuple)) else 1
    geo = {'source_tokens': len(sids), 'image_tokens': 256 * count, 'prefix_tokens': len(rlist),
           'caption_query_positions': sum(b > rstart and a < rstart+len(caption) for a,b in roffsets)}
    visual = np.flatnonzero(np.array(sids) == processor.image_token_id)
    count = len(image) if isinstance(image, (list, tuple)) else 1
    assert len(visual) == 256 * count and tuple(batch['pixel_values'].shape) == (count, 3, 896, 896)
    keep = content_mask | (np.arange(len(sids)) < visual[0])
    assert not batch['token_type_ids'][0, :visual[0]].any()
    assert not keep[visual].any()
    assert not np.any(keep & (np.arange(len(sids)) >= visual[0]) & ~content_mask)
    allowed = (np.arange(len(sids))[None, :] < limits[0].numpy()[:, None]) & keep[None, :]
    assert not allowed[:, visual].any()
    geo.update(selected_language_tokens=int(content_mask.sum()), retained_header_tokens=int((keep & ~content_mask).sum()),
               permitted_source_positions=np.flatnonzero(keep).tolist(), max_cross_keys=int(allowed.sum(1).max()))
    return batch, rids, mpos, limits, torch.tensor(keep[None]), geo


def image_content(image, prompt, numbered=False):
    images = image if isinstance(image, (list, tuple)) else [image]
    content = []
    for i in range(len(images)):
        if numbered and len(images) > 1:
            content.append({'type': 'text', 'text': f'<image {i + 1}>'})
        content.append({'type': 'image'})
    return content + [{'type': 'text', 'text': prompt}]


def caption_inputs(processor, image, prompt, numbered=False):
    text = processor.apply_chat_template(
        [{'role': 'user', 'content': image_content(image, prompt, numbered)}],
        tokenize=False, add_generation_prompt=False)
    batch = processor(text=text, images=image, add_special_tokens=False,
                      return_tensors='pt', do_pan_and_scan=False)
    count = len(image) if isinstance(image, (list, tuple)) else 1
    assert int((batch['input_ids'] == processor.image_token_id).sum()) == 256 * count
    assert tuple(batch['pixel_values'].shape) == (count, 3, 896, 896)
    return batch


def gemma_followup_inputs(processor, image, prompt, assistant, followup):
    marker = '__GEMMA_REPLAY_SENTINEL__'
    if marker in prompt or marker in followup:
        raise ValueError('Replay sentinel occurs in user text')
    messages = [{'role': 'user', 'content': image_content(image, prompt)},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': marker}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': followup}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    before, after = rendered.split(marker)
    if isinstance(assistant, str):
        batch = processor(text=before + assistant + after, images=image, add_special_tokens=False,
                          return_tensors='pt', do_pan_and_scan=False)
    else:
        batch = processor(text=before, images=image, add_special_tokens=False,
                          return_tensors='pt', do_pan_and_scan=False)
        suffix = processor.tokenizer(after, add_special_tokens=False)['input_ids']
        extra = torch.tensor([list(assistant) + suffix], dtype=batch['input_ids'].dtype)
        batch['input_ids'] = torch.cat((batch['input_ids'], extra), dim=1)
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['token_type_ids'] = torch.cat((batch['token_type_ids'], torch.zeros_like(extra)), dim=1)
    count = len(image) if isinstance(image, (list, tuple)) else 1
    assert int((batch['input_ids'] == processor.image_token_id).sum()) == 256 * count
    assert tuple(batch['pixel_values'].shape) == (count, 3, 896, 896)
    return batch


class Engine:
    @classmethod
    def baseliner(cls, sender, receiver, methods, device='cuda:0'):
        self = cls.captioner(sender, device) if set(methods) & set(GEMMA_BASELINES) else cls.__new__(cls)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        self.device = device
        if set(methods) & {'granite_only', 't2t', 't2tplus'}:
            self.tok = AutoTokenizer.from_pretrained(receiver, local_files_only=True)
            self.system_date_applied = freeze_system_date(self.tok)
            self.receiver = load_causal_model(Path(receiver), torch.bfloat16, device).eval().requires_grad_(False)
        return self

    @torch.inference_mode()
    def baseline_answer(self, image, question, method, caption=None, ocr=None, reasoning_budget=None):
        started = time.monotonic()
        metadata = {}
        if method in GEMMA_BASELINES:
            if method == 'gemma_only':
                batch = caption_inputs(self.processor, image, question + '\n' + INSTRUCTION)
            elif method == 'gemma_cot':
                if type(reasoning_budget) is not int or not 1 <= reasoning_budget <= 352:
                    raise ValueError('Gemma CoT requires the saved S+OCR token budget (1..352)')
                prompt = question + '\n' + REASONING_PROMPT
                reasoning = self.describe(image, prompt, reasoning_budget)
                batch = gemma_followup_inputs(self.processor, image, prompt, reasoning['output_ids'], INSTRUCTION)
                metadata = {'reasoning': reasoning, 'reasoning_budget': reasoning_budget}
            else:
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError('Gemma S/ST requires a saved caption')
                if method == 'gemma_st':
                    if not isinstance(ocr, str):
                        raise ValueError('Gemma ST requires saved OCR (empty text is allowed)')
                    if ocr:
                        caption += '\n\nAdditional visible text:\n' + ocr
                batch = gemma_followup_inputs(self.processor, image, CAPTION_PROMPT, caption,
                                              question + '\n' + INSTRUCTION)
            batch = on_device(batch, self.device)
            ids = batch['input_ids']
            cache = self.sender(**batch,
                position_ids=torch.arange(ids.shape[1], device=self.device)[None],
                past_key_values=full_sender_cache(), use_cache=True).past_key_values
            answer = decode(self.sender, self.processor.tokenizer, cache, ids.shape[1], 32, self.device)
        elif method in ('granite_only', 't2t', 't2tplus'):
            if method in ('t2t', 't2tplus') and not caption:
                raise ValueError('T2T requires a saved caption')
            if method == 't2tplus':
                if not isinstance(ocr, str):
                    raise ValueError('T2T+ requires saved OCR text (empty text is allowed)')
                if ocr:
                    caption += '\n\nAdditional visible text:\n' + ocr
            ids = receiver_ids(self.tok, question, caption if method != 'granite_only' else None).to(self.device)
            cache = self.receiver(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True).past_key_values
            answer = decode(self.receiver, self.tok, cache, ids.shape[1], 32, self.device)
        else:
            raise ValueError(f'Unknown baseline: {method}')
        return {**answer, **metadata, 'input_ids': ids[0].tolist(), 'seconds': time.monotonic() - started}

    @torch.inference_mode()
    def baseline_batch(self, items, method):
        if method not in GEMMA_BASELINES or not items:
            raise ValueError('A nonempty batch of one Gemma method is required')
        if len(items) == 1:
            result = self.baseline_answer(method=method, **items[0])
            return [{**result, 'batch_size': 1, 'batch_seconds': result['seconds']}]
        try:
            return self._baseline_batch(items, method)
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        import sys
        print(f'{method}: CUDA OOM at batch size {len(items)}; retrying smaller batches', file=sys.stderr, flush=True)
        mid = len(items) // 2
        return self.baseline_batch(items[:mid], method) + self.baseline_batch(items[mid:], method)

    def _gemma_batch_generate(self, inputs, budgets):
        batch = on_device(pad_gemma_inputs(inputs, self.processor.tokenizer.pad_token_id), self.device)
        positions = (batch['attention_mask'].cumsum(dim=1) - 1).clamp(min=0)
        cache = self.sender(**batch, position_ids=positions, past_key_values=full_sender_cache(),
                            use_cache=True, logits_to_keep=1).past_key_values
        return decode_batch(self.sender, self.processor.tokenizer, cache, batch['attention_mask'], budgets, self.device)

    def _baseline_batch(self, items, method):
        started = time.monotonic()
        metadata = [{} for _ in items]
        if method == 'gemma_cot':
            budgets = [item.get('reasoning_budget') for item in items]
            if any(type(n) is not int or not 1 <= n <= 352 for n in budgets):
                raise ValueError('Gemma CoT requires saved S+OCR token budgets (1..352)')
            prompts = [item['question'] + '\n' + REASONING_PROMPT for item in items]
            inputs = [caption_inputs(self.processor, item['image'], prompt) for item, prompt in zip(items, prompts)]
            reasoning = self._gemma_batch_generate(inputs, budgets)
            metadata = [{'reasoning': {**r, 'sender_input_sha256': digest_tensor(batch['input_ids'])},
                         'reasoning_budget': budget} for r, batch, budget in zip(reasoning, inputs, budgets)]
            inputs = [gemma_followup_inputs(self.processor, item['image'], prompt, r['output_ids'], INSTRUCTION)
                      for item, prompt, r in zip(items, prompts, reasoning)]
        else:
            inputs = []
            for item in items:
                if method == 'gemma_only':
                    batch = caption_inputs(self.processor, item['image'], item['question'] + '\n' + INSTRUCTION)
                else:
                    caption = item.get('caption')
                    if not isinstance(caption, str) or not caption.strip():
                        raise ValueError('Gemma S/ST requires a saved caption')
                    if method == 'gemma_st':
                        ocr = item.get('ocr')
                        if not isinstance(ocr, str):
                            raise ValueError('Gemma ST requires saved OCR (empty text is allowed)')
                        if ocr:
                            caption += '\n\nAdditional visible text:\n' + ocr
                    batch = gemma_followup_inputs(self.processor, item['image'], CAPTION_PROMPT, caption,
                                                  item['question'] + '\n' + INSTRUCTION)
                inputs.append(batch)
        answers = self._gemma_batch_generate(inputs, [32] * len(items))
        elapsed = time.monotonic() - started
        return [{**answer, **meta, 'input_ids': batch['input_ids'][0].tolist(),
                 'seconds': elapsed / len(items), 'batch_seconds': elapsed, 'batch_size': len(items)}
                for answer, meta, batch in zip(answers, metadata, inputs)]

    @classmethod
    def captioner(cls, sender, device='cuda:0'):
        self = cls.__new__(cls)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(sender, local_files_only=True)
        self.sender = Gemma3ForConditionalGeneration.from_pretrained(
            sender, dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
        return self

    def __init__(self, sender, receiver, renderer, device='cuda:0'):
        torch.manual_seed(42)
        torch.set_num_threads(4)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(sender, local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(receiver, local_files_only=True)
        self.system_date_applied = freeze_system_date(self.tok)
        self.renderer, payload = load_renderer_checkpoint(Path(renderer), device)
        if payload['step'] != 24000:
            raise ValueError('Expected renderer checkpoint step 24000')
        del payload
        self.sender = Gemma3ForConditionalGeneration.from_pretrained(
            sender, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        self.receiver = load_causal_model(Path(receiver), torch.bfloat16, device)
        for model in (self.sender, self.receiver, self.renderer):
            model.eval().requires_grad_(False)

    @torch.inference_mode()
    def describe(self, image, prompt=CAPTION_PROMPT, budget=CAPTION_BUDGET, numbered=False):
        batch = on_device(caption_inputs(self.processor, image, prompt, numbered), self.device)
        pos = torch.arange(batch['input_ids'].shape[1], device=self.device)[None]
        cache = self.sender(**batch, position_ids=pos, past_key_values=full_sender_cache(),
                            use_cache=True).past_key_values
        result = decode(self.sender, self.processor.tokenizer, cache, pos.shape[1], budget, self.device)
        return {**result, 'sender_input_sha256': digest_tensor(batch['input_ids'])}

    @torch.inference_mode()
    def describe_batch(self, images, budget=CAPTION_BUDGET):
        if not images:
            raise ValueError('A nonempty batch of image bundles is required')
        if len(images) == 1:
            return [{**self.describe(images[0], budget=budget), 'batch_size': 1}]
        inputs = [caption_inputs(self.processor, image, CAPTION_PROMPT) for image in images]
        results = None
        try:
            results = self._gemma_batch_generate(inputs, [budget] * len(inputs))
        except torch.cuda.OutOfMemoryError:
            pass
        if results is None:
            torch.cuda.empty_cache()
            import sys
            print(f'captions: CUDA OOM at batch size {len(images)}; retrying smaller batches', file=sys.stderr, flush=True)
            mid = len(images) // 2
            return self.describe_batch(images[:mid], budget) + self.describe_batch(images[mid:], budget)
        return [{**result, 'sender_input_sha256': digest_tensor(batch['input_ids']), 'batch_size': len(images)}
                for result, batch in zip(results, inputs)]

    @torch.inference_mode()
    def transcribe(self, image):
        saved = self.describe(image, OCR_PROMPT, 64, numbered=True)
        if saved['stop_reason'] == 'eos':
            return saved
        assert len(saved['output_ids']) == 64
        batch = on_device(caption_inputs(self.processor, image, OCR_PROMPT, numbered=True), self.device)
        assert digest_tensor(batch['input_ids']) == saved['sender_input_sha256']
        pos = batch['input_ids'].shape[1]
        cache = self.sender(**batch, position_ids=torch.arange(pos, device=self.device)[None],
                            past_key_values=full_sender_cache(), use_cache=True).past_key_values
        tokens = torch.tensor([generation_prompt_ids(self.processor.tokenizer) + saved['output_ids']],
                              device=self.device)
        stop = turn_end_ids(self.processor.tokenizer, self.sender.generation_config)
        generated, reason = list(saved['output_ids']), 'cutoff'
        for _ in range(32):
            output = self.sender(input_ids=tokens, past_key_values=cache, use_cache=True,
                attention_mask=torch.ones((1, pos + tokens.shape[1]), device=self.device, dtype=torch.long),
                position_ids=torch.arange(pos, pos + tokens.shape[1], device=self.device)[None])
            cache = output.past_key_values
            pos += tokens.shape[1]
            next_id = int(output.logits[0, -1].argmax())
            if next_id in stop:
                reason = 'eos'
                break
            generated.append(next_id)
            tokens = torch.tensor([[next_id]], device=self.device)
        return {**saved, 'output': self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'output_ids': generated, 'stop_reason': reason, 'continued_from_tokens': 64}

    @torch.inference_mode()
    def prefix(self, image, caption, enhanced=False, value_scale=1., memo=None):
        batch, rids, mpos, limits, mask, geo = caption_layout(self.processor, self.tok, image, caption)
        visual = batch['input_ids'][0] == self.processor.image_token_id
        if enhanced:
            mask[0, torch.nonzero(visual).flatten()[::16]] = True
            start = min(p for p in geo['permitted_source_positions'] if p >= geo['retained_header_tokens'])
            queries = limits[0] > start
            limits[0, queries] = (limits[0, queries] + 32).clamp(max=mask.shape[1])
        geo.update(selected_source_positions=torch.nonzero(mask[0]).flatten().tolist(),
                   selected_image_tokens=int(mask[0, visual].sum()),
                   visibility='lookahead32' if enhanced else 'original_causal',
                   source_selection='summary_header',
                   memory_positions_sha256=digest_tensor(mpos), limits_sha256=digest_tensor(limits))
        batch = on_device(batch, self.device)
        key = tuple(batch['input_ids'][0].tolist())
        if memo is not None and key in memo:
            memory = memo[key]
        else:
            spos = torch.arange(len(key), device=self.device)[None]
            cache = self.sender(**batch, position_ids=spos, past_key_values=full_sender_cache(),
                                use_cache=True).past_key_values
            memory = flatten_sender_cache(self.sender, cache, spos)
            if memo is not None:
                memo[key] = memory
        rids = rids.to(self.device)
        rpos = torch.arange(rids.shape[1], device=self.device)[None]
        mpos, limits, mask = (x.to(self.device) for x in (mpos, limits, mask))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = self.renderer(self.receiver.get_input_embeddings()(rids), rpos, memory, mpos, mask, limits)
        assert torch.isfinite(pred).all()
        kv = pred_to_cache(self.receiver, pred, rpos, torch.bfloat16)
        if value_scale != 1.:
            kv = tuple(((k.float() * 1.).to(k.dtype), (v.float() * value_scale).to(v.dtype)) for k, v in kv)
        info = {'receiver_prefix_ids': rids[0].tolist(), 'geometry': geo,
                'rendered_sha256': digest_tensor(pred), 'sender_input_sha256': digest_json(list(key)),
                'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(),
                'transferred_sha256': digest_json([[digest_tensor(k), digest_tensor(v)] for k, v in kv])}
        return kv, info

    @torch.inference_mode()
    def prepare(self, image, methods=VOTERS, caption=None, ocr=None):
        if not methods or set(methods) - set(VOTERS):
            raise ValueError('Unknown or empty voter selection')
        caption = self.describe(image) if caption is None else dict(caption)
        if not caption['output']:
            raise ValueError('Gemma generated an empty caption')
        text = caption['output']
        ocr = (self.transcribe(image) if ocr is None else dict(ocr)) if 'textscan_v6' in methods else None
        extra_text = text
        if ocr and ocr['output']:
            extra_text += '\n\nAdditional visible text:\n' + ocr['output']
        prefixes, memo = {}, {}
        if 'captkv' in methods or 'echo_v1' in methods:
            shared = self.prefix(image, text, memo=memo)
            for method in ('captkv', 'echo_v1'):
                if method in methods:
                    prefixes[method] = (*shared, text)
        if 'kvgain_v3' in methods:
            prefixes['kvgain_v3'] = (*self.prefix(image, text, True, 1.3, memo), text)
        if 'textscan_v6' in methods:
            prefixes['textscan_v6'] = (*self.prefix(image, extra_text, True, memo=memo), extra_text)
        return {'caption': caption, 'ocr': ocr, 'prefixes': prefixes}

    @torch.inference_mode()
    def answer(self, prepared, question):
        predictions = {}
        for method, (kv, info, caption) in prepared['prefixes'].items():
            started = time.monotonic()
            text = question
            if method != 'captkv':
                text = 'Image description:\n' + caption + '\n\nQuestion:\n' + question
            ids = receiver_ids(self.tok, text, caption).to(self.device)
            length = len(info['receiver_prefix_ids'])
            assert ids[0, :length].tolist() == info['receiver_prefix_ids']
            cache = tuple_to_dynamic_cache(tuple((k.clone(), v.clone()) for k, v in kv), self.receiver.config)
            cache = self.receiver(input_ids=ids[:, length:], past_key_values=cache,
                attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(length, ids.shape[1], device=self.device)[None],
                use_cache=True).past_key_values
            result = decode(self.receiver, self.tok, cache, ids.shape[1], 32, self.device)
            predictions[method] = {**result, 'prefix_sha256': info['transferred_sha256'],
                                   'receiver_input_sha256': digest_json(ids[0].tolist()),
                                   'receiver_input_ids': ids[0].tolist(), 'seconds': time.monotonic() - started}
        return predictions
