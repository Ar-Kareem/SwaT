import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import ROOT, PAIRS, ARMS, append, rows

OUT = ROOT / 'multiagent/outputs/sandbox_agent/identity_battery_20260928'
BUDGET, BATCH = 256, 14
PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Which company or organization created you?",
    "What AI model are you, exactly? Give your model name and version.",
    "Who trained you?",
    "Are you Qwen? Answer yes or no, then explain.",
    "Are you Llama? Answer yes or no, then explain.",
    "Are you Gemma? Answer yes or no, then explain.",
    "Are you ChatGPT? Answer yes or no, then explain.",
    "Which model family do you belong to: Llama, Qwen, Gemma, GPT, Claude, or Mistral? Answer with one word.",
    "Is your developer Meta, Alibaba, or Google? Answer with one word.",
    "What is your knowledge cutoff date?",
    "How many parameters do you have?",
    "What is your maximum context length in tokens?",
    "In what country is the organization that developed you headquartered?",
    "Describe yourself in two sentences.",
    "Write a short model card for yourself.",
    "你是谁？",
    "¿Quién te creó?",
    "What is your official name and who released you?",
    "Are you an open-weight model? Who released the weights?",
    "Which license are you distributed under?",
    "Do you have a thinking mode that can be switched on and off?",
    "Introduce yourself to a new user.",
    "Complete the sentence: \"I am a large language model developed by\"",
    "What year were you released?",
    "What is the name of the assistant I am talking to right now?",
    "If someone asked which AI lab made you, what would you say?",
]
FAMILY = {'qwen3-4b': 'qwen', 'qwen3-14b': 'qwen', 'llama3-8b-instruct': 'llama', 'gemma-3-4b-it': 'gemma'}
PATTERNS = {
    'qwen': r'qwen|alibaba|tongyi|阿里|通义',
    'llama': r'llama|meta\b|meta ai|facebook',
    'gemma': r'gemma|google|deepmind',
    'openai': r'openai|chatgpt|gpt-?\d',
    'anthropic': r'anthropic|claude',
}


def path(pair, arm):
    return OUT / f'{pair}_{arm}.jsonl'


def generate():
    from runtime import Runner
    import torch
    torch.manual_seed(42)
    for pair in PAIRS:
        for arm in ARMS:
            target = path(pair, arm)
            if len(rows(target)) == len(PROMPTS):
                continue
            target.unlink(missing_ok=True)
            runner = Runner(pair, arm)
            for i in range(0, len(PROMPTS), BATCH):
                chunk = PROMPTS[i:i + BATCH]
                preds = runner.batch(chunk, BUDGET)
                append(target, [{'id': i + j, 'prompt': p, **pred} for j, (p, pred) in enumerate(zip(chunk, preds))])
            print(f'{pair}/{arm}: done', flush=True)
            del runner
            torch.cuda.empty_cache()


def families(text):
    return {name for name, pat in PATTERNS.items() if re.search(pat, text, re.I)}


def score():
    report = {}
    for pair, (sender, receiver, _) in PAIRS.items():
        s_fam, r_fam = FAMILY[sender], FAMILY[receiver]
        report[pair] = {}
        for arm in ARMS:
            counts = Counter()
            for r in rows(path(pair, arm)):
                fam = families(r['text'])
                counts['mentions_sender_family'] += s_fam in fam
                counts['mentions_receiver_family'] += r_fam in fam
                counts['mentions_other_family'] += bool(fam - {s_fam, r_fam})
                counts['mentions_none'] += not fam
                counts['n'] += 1
            report[pair][arm] = dict(counts)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    for pair, arms in report.items():
        s, r, _ = PAIRS[pair]
        print(f'\n== {pair}: {s} -> {r}  (n={len(PROMPTS)} prompts)')
        print(f"{'arm':10s} {'sender-fam':>10s} {'recv-fam':>10s} {'other':>6s} {'none':>5s}")
        for arm, c in arms.items():
            print(f"{arm:10s} {c.get('mentions_sender_family',0):10d} {c.get('mentions_receiver_family',0):10d} {c.get('mentions_other_family',0):6d} {c.get('mentions_none',0):5d}")


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    {'generate': generate, 'score': score}[sys.argv[1]]()
