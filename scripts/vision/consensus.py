from collections import Counter
import re

VOTERS = ("textscan_v6", "captkv", "echo_v1", "kvgain_v3")


def canonical(text):
    text = re.sub(r"[^\w.%+-]+", " ", text.casefold()).strip().rstrip(".!?")
    words = text.split()
    if len(words) > 1:
        words = [w for w in words if w not in ("a", "an", "the")]
    numbers = dict(zip(("zero one two three four five six seven eight nine ten eleven twelve "
                        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(),
                       map(str, range(21))))
    return " ".join(numbers.get(w, w) for w in words)


def vote_index(outputs, weights=None):
    normalized = [canonical(s) for s in outputs]
    counts = Counter(normalized)

    def support(ws):
        return [sum(w for b, w in zip(normalized, ws)
                    if a and (" " + a + " ") in (" " + b + " ")) for a in normalized]

    def best(scores):
        return min(range(len(outputs)), key=lambda i:
                   (-scores[i], len(outputs[i].split()), len(outputs[i].strip()), i))

    if len(outputs) == 3:
        scores = support([1] * 3)
        if max(scores) > max(counts.values()):
            return best(scores)
        if max(counts.values()) == 1:
            return min(range(3), key=lambda i:
                       (not bool(outputs[i].strip()), len(outputs[i].split()), len(outputs[i].strip()), i))
        return best([counts[s] for s in normalized])
    if len(outputs) != 4:
        raise ValueError("ScanTrust requires four voter answers")
    anchor = vote_index(outputs[:3])
    scores = support([1] * 4)
    if max(scores) > scores[anchor]:
        anchor = best(scores)
    if weights is not None:
        scores = support(weights)
        if max(scores) > scores[anchor] + 1e-9:
            anchor = best(scores)
    return anchor


def consensus(predictions, ocr_tokens):
    missing = set(VOTERS) - predictions.keys()
    if missing:
        raise ValueError(f"Missing voters: {', '.join(sorted(missing))}")
    if type(ocr_tokens) is not int or ocr_tokens < 0:
        raise ValueError("The TextScan OCR token count is required for ScanTrust/v2")
    outputs = [predictions[m]["output"] for m in VOTERS]
    weights = [4 if ocr_tokens >= 64 else 1, 1, 1, 1]
    winner = vote_index(outputs, weights)
    return {"method": "scantrust_v2", "output": outputs[winner],
            "selected_voter": VOTERS[winner], "weights": dict(zip(VOTERS, weights))}
