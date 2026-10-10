from experiment import MODELS, PAIRS, ARMS, checkpoint
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from multiagent.dataset.c2c import apply_chat_template, generation_prompt_ids, turn_end_ids
from multiagent.dataset.c2c_cross_family import encode_context_pair
from multiagent.model.cache_renderer import flatten_sender_cache, load_renderer_checkpoint, pred_to_cache
from multiagent.model.unaware_comm import full_sender_cache, tuple_to_dynamic_cache
from multiagent.model.vanilla_loader import load_causal_model, resolve_model_path


def tokenizer_for(name):
    tok = AutoTokenizer.from_pretrained(resolve_model_path(name, MODELS), local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok


def messages(prompt):
    return [{"role": "user", "content": prompt}]


def prompt_ids(tok, prompt, generation):
    text = apply_chat_template(tok, messages(prompt), add_generation_prompt=generation)
    return tok(text, add_special_tokens=False)["input_ids"]


def left_pad(rows, pad_id, device):
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, -len(row):] = torch.tensor(row, device=device)
        mask[i, -len(row):] = 1
    return ids, mask


@torch.inference_mode()
def decode(model, tok, ids, mask, positions, cache, budget):
    stop = turn_end_ids(tok, model.generation_config)
    active = [True] * ids.shape[0]
    generated = [[] for _ in active]
    reasons = ["cutoff"] * len(active)
    lengths = mask.sum(1)
    for step in range(budget):
        result = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        chosen = result.logits[:, -1].argmax(-1).tolist()
        del result
        for i, token in enumerate(chosen):
            if active[i]:
                if token in stop:
                    reasons[i], active[i] = "eos", False
                else:
                    generated[i].append(token)
        if not any(active):
            break
        ids = torch.tensor([[t if a else tok.pad_token_id] for t, a in zip(chosen, active)], device=mask.device)
        mask = torch.cat([mask, torch.tensor(active, dtype=mask.dtype, device=mask.device)[:, None]], 1)
        positions = (lengths + step)[:, None]
    return [{"text": tok.decode(t, skip_special_tokens=True).strip(),
             "tokens": len(t), "stop_reason": r} for t, r in zip(generated, reasons)]


class Runner:
    def __init__(self, pair, arm, device='cuda:0'):
        assert arm in ARMS, "Unsupported experiment arm"
        self.pair, self.arm, self.device = pair, arm, device
        sender, receiver, _ = PAIRS[pair]
        self.sender_tok = tokenizer_for(sender)
        self.receiver_tok = tokenizer_for(receiver)
        self.tok = self.sender_tok if arm == 'sender' else self.receiver_tok
        self.renderer = None
        if arm == 'renderer':
            self.renderer, payload = load_renderer_checkpoint(checkpoint(pair), device)
            assert payload['sender_model'].lower() == sender
            assert payload['receiver_model'].lower() == receiver
            assert self.renderer.token_alignment == 'native'
            assert payload['args']['causal_cross_attention']
            self.position_mode = payload.get('position_mode') or 'interp'
            self.sender = load_causal_model(resolve_model_path(sender, MODELS), torch.bfloat16, device)
            del payload
        self.model = load_causal_model(resolve_model_path(sender if arm == 'sender' else receiver,
                                                                    MODELS), torch.bfloat16, device)
        self.text_config = self.model.config.get_text_config()
        for model in (self.model, getattr(self, 'sender', None), self.renderer):
            if model is not None:
                model.eval().requires_grad_(False)

    @torch.inference_mode()
    def render(self, prompt):
        context = encode_context_pair(self.sender_tok, self.receiver_tok, messages(prompt), 8192, self.position_mode)
        assert context["receiver_ids"] == prompt_ids(self.receiver_tok, prompt, False), "Receiver prompt truncated"
        assert context["sender_ids"] == prompt_ids(self.sender_tok, prompt, False), "Sender prompt truncated"
        sids = torch.tensor([context["sender_ids"]], device=self.device)
        spos = torch.arange(sids.shape[1], device=self.device)[None]
        scache = self.sender(input_ids=sids, attention_mask=torch.ones_like(sids), position_ids=spos,
                             use_cache=True, past_key_values=full_sender_cache(), logits_to_keep=1).past_key_values
        memory = flatten_sender_cache(self.sender, scache, spos)
        del scache
        rids = torch.tensor([context["receiver_ids"]], device=self.device)
        rpos = torch.arange(rids.shape[1], device=self.device)[None]
        virtual = torch.as_tensor(context["virtual_positions"], device=self.device, dtype=torch.float32)[None]
        limits = torch.as_tensor(context["cross_causal_limits"], device=self.device)[None]
        embeds = self.model.get_input_embeddings()(rids)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.renderer(embeds, rpos, memory, virtual, None, limits)
        return pred_to_cache(self.model, pred, rpos, torch.bfloat16)

    def validate_prompts(self, prompts, budget):
        header = generation_prompt_ids(self.tok)
        limit = self.text_config.max_position_embeddings
        for prompt in prompts:
            prefix = prompt_ids(self.tok, prompt, False)
            complete = prompt_ids(self.tok, prompt, True)
            assert prefix + header == complete, 'Cache split differs from full template'
            assert len(complete) + budget <= limit, (len(complete), budget, limit)
        return header

    @torch.inference_mode()
    def decode_caches(self, prompts, caches, budget):
        header = self.validate_prompts(prompts, budget)
        lengths = [c[0][0].shape[2] for c in caches]
        for prompt, cache, n in zip(prompts, caches, lengths):
            assert n == len(prompt_ids(self.tok, prompt, False))
            assert len(cache) == self.text_config.num_hidden_layers
            assert all(k.shape[2] == v.shape[2] == n for k, v in cache)
        width = max(lengths)
        packed = tuple(tuple(torch.cat([F.pad(c[layer][kv], (0, 0, width - n, 0))
                                         for c, n in zip(caches, lengths)], 0)
                             for kv in (0, 1)) for layer in range(len(caches[0])))
        cache = tuple_to_dynamic_cache(packed, self.model.config)
        mask = torch.zeros((len(prompts), width + len(header)), dtype=torch.long, device=self.device)
        for i, length in enumerate(lengths):
            mask[i, width - length:] = 1
        ids = torch.tensor([header] * len(prompts), device=self.device)
        positions = torch.tensor(lengths, device=self.device)[:, None] + torch.arange(len(header), device=self.device)[None]
        return decode(self.model, self.tok, ids, mask, positions, cache, budget)

    @torch.inference_mode()
    def batch(self, prompts, budget):
        self.validate_prompts(prompts, budget)
        if self.arm == 'renderer':
            caches = [self.render(p) for p in prompts]
            return self.decode_caches(prompts, caches, budget)
        ids, mask = left_pad([prompt_ids(self.tok, p, True) for p in prompts], self.tok.pad_token_id, self.device)
        return decode(self.model, self.tok, ids, mask, (mask.cumsum(1) - 1).clamp_min(0), None, budget)
