from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


SCHEDULER_CHOICES = ("constant", "linear", "cosine")


def add_lr_scheduler_args(parser) -> None:
    parser.add_argument(
        "--lr-scheduler",
        choices=SCHEDULER_CHOICES,
        default="constant",
        help="Learning-rate schedule. The default preserves the previous constant-LR behavior.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of optimizer steps used for linear LR warmup.",
    )
    parser.add_argument(
        "--lr-total-steps",
        type=int,
        help="Optimizer-step horizon for linear/cosine decay. Defaults to --max-steps or the inferred run length.",
    )


def resolve_total_steps(args, loader_length: int, optimizer_steps_per_batch: int = 1) -> tuple[int, str]:
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.lr_total_steps is not None and args.lr_total_steps <= 0:
        raise ValueError("--lr-total-steps must be positive")
    if args.lr_total_steps is not None:
        return args.lr_total_steps, "explicit --lr-total-steps"
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        return args.max_steps, "--max-steps"
    total_steps = max(1, args.epochs * loader_length * optimizer_steps_per_batch)
    source = "epochs * dataloader batches"
    if optimizer_steps_per_batch != 1:
        source += f" * {optimizer_steps_per_batch} estimated sub-batch steps"
    return total_steps, source


def lr_multiplier(schedule: str, warmup_steps: int, total_steps: int, completed_steps: int) -> float:
    if warmup_steps > 0 and completed_steps < warmup_steps:
        return float(completed_steps + 1) / float(warmup_steps)
    if schedule == "constant":
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (completed_steps - warmup_steps) / decay_steps))
    if schedule == "linear":
        return 1.0 - progress
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown LR scheduler: {schedule}")


def build_lr_scheduler(optimizer, schedule: str, warmup_steps: int, total_steps: int) -> LambdaLR:
    if schedule not in SCHEDULER_CHOICES:
        raise ValueError(f"unknown LR scheduler: {schedule}")
    if warmup_steps > total_steps:
        raise ValueError(
            f"--warmup-steps ({warmup_steps}) cannot exceed the LR schedule horizon ({total_steps})"
        )
    return LambdaLR(
        optimizer,
        lr_lambda=lambda completed_steps: lr_multiplier(
            schedule,
            warmup_steps,
            total_steps,
            completed_steps,
        ),
    )


def restore_lr_scheduler(scheduler: LambdaLR, state: dict | None, completed_steps: int) -> None:
    if state is not None:
        scheduler.load_state_dict(state)
        return

    scheduler.last_epoch = completed_steps
    scheduler._step_count = completed_steps + 1
    lrs = [base_lr * fn(completed_steps) for base_lr, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs):
        param_group["lr"] = lr
    scheduler._last_lr = lrs
