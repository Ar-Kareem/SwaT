from pathlib import Path

from audit_vision_test import write_audit


if __name__ == "__main__":
    write_audit("vision_val", Path(__file__).resolve().with_suffix(".jsonl"))
