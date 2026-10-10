#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="outputs/head_pairings/features.pt")
    parser.add_argument("--output-dir", help="Default: directory of --features.")
    parser.add_argument("--tie-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="Force K and V of a receiver unit to share one sender unit (default on).")
    parser.add_argument("--rms-scales", action=argparse.BooleanOptionalAction, default=False,
                        help="Normalize each unit by its train-split RMS before fitting (default off).")
    parser.add_argument("--allowed-width", type=int,
                        help="Restrict assignment to a window of this many sender layers centered on the "
                             "proportional layer map (3 -> +/-1, 5 -> +/-2, ...). Default: unrestricted.")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3,
                        help="Ridge strength relative to each sender unit's mean feature energy.")
    parser.add_argument("--heldout-fraction", type=float, default=0.25, help="Fraction of documents held out.")
    parser.add_argument("--receiver-chunk", type=int, default=32, help="Receiver units solved per chunk.")
    parser.add_argument("--seed", type=int, default=0, help="Document split shuffle seed.")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def flat_units(tensor: torch.Tensor, device: str) -> torch.Tensor:
    n, layers, heads, dim = tensor.shape
    return tensor.reshape(n, layers * heads, dim).to(device=device, dtype=torch.float32)


def augment(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_ones(x.shape[:-1] + (1,))], dim=-1)


def unit_gram(x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nsd,nse->sde", x, x)


@torch.no_grad()
def ridge_scores(x_train, y_train, x_ho, y_ho, ridge_lambda: float, chunk: int):
    n_train = x_train.shape[0]
    s_units, dim_x = x_train.shape[1], x_train.shape[2]
    r_units = y_train.shape[1]

    xa_train = augment(x_train)
    xa_ho = augment(x_ho)
    gram = unit_gram(xa_train)
    gram_ho = unit_gram(xa_ho)
    y_energy_ho = y_ho.float().pow(2).sum(dim=(0, 2))

    mean_energy = gram.diagonal(dim1=-2, dim2=-1)[:, :dim_x].mean(dim=-1)
    penalty = torch.zeros(dim_x + 1, device=gram.device)
    penalty[:dim_x] = 1.0
    lhs = gram + ridge_lambda * mean_energy[:, None, None] * torch.diag(penalty)

    sum_x = gram[:, :dim_x, dim_x]
    sum_y = y_train.sum(dim=0)
    gram_x_centered = gram[:, :dim_x, :dim_x] - sum_x[:, :, None] * sum_x[:, None, :] / n_train
    gram_y = unit_gram(y_train)
    gram_y_centered = gram_y - sum_y[:, :, None] * sum_y[:, None, :] / n_train
    norm_x = gram_x_centered.flatten(1).norm(dim=1).clamp_min(1e-12)
    norm_y = gram_y_centered.flatten(1).norm(dim=1).clamp_min(1e-12)

    rel_mse = torch.empty(s_units, r_units)
    cka = torch.empty(s_units, r_units)
    for start in range(0, r_units, chunk):
        stop = min(start + chunk, r_units)
        cross = torch.einsum("nsd,nre->srde", xa_train, y_train[:, start:stop])
        weights = torch.linalg.solve(lhs[:, None], cross)
        cross_ho = torch.einsum("nsd,nre->srde", xa_ho, y_ho[:, start:stop])
        fit = torch.einsum("scde,scde->sc", weights, cross_ho)
        quad = torch.einsum("scde,sdf,scfe->sc", weights, gram_ho, weights)
        sse = (y_energy_ho[start:stop][None, :] - 2.0 * fit + quad).clamp_min(0.0)
        rel_mse[:, start:stop] = (sse / y_energy_ho[start:stop].clamp_min(1e-12)[None, :]).cpu()

        cross_centered = cross[:, :, :dim_x, :] - sum_x[:, None, :, None] * sum_y[start:stop][None, :, None, :] / n_train
        hsic = cross_centered.pow(2).sum(dim=(-2, -1))
        cka[:, start:stop] = (hsic / (norm_x[:, None] * norm_y[start:stop][None, :])).cpu()
        del cross, weights, cross_ho, cross_centered
    return rel_mse, cka


def proportional_map(sender_units: int, receiver_units: int) -> list[int]:
    if receiver_units <= 1:
        return [0]
    sender_last = max(sender_units - 1, 0)
    receiver_last = receiver_units - 1
    return [round(i * sender_last / receiver_last) for i in range(receiver_units)]


def baseline_assignment(sender_shape, receiver_shape) -> torch.Tensor:
    sender_layers, sender_heads, _ = sender_shape
    receiver_layers, receiver_heads, _ = receiver_shape
    layer_map = proportional_map(sender_layers, receiver_layers)
    group_map = proportional_map(sender_heads, receiver_heads)
    rows = [layer_map[l] * sender_heads + group_map[h] for l in range(receiver_layers) for h in range(receiver_heads)]
    return torch.tensor(rows, dtype=torch.long)


def allowed_mask(sender_shape, receiver_shape, base: torch.Tensor, width: int | None) -> torch.Tensor:
    s_units = sender_shape[0] * sender_shape[1]
    if width is None:
        return torch.ones(s_units, base.numel(), dtype=torch.bool)
    if width < 1:
        raise ValueError("--allowed-width must be >= 1")
    sender_layer = torch.arange(s_units) // sender_shape[1]
    base_layer = base // sender_shape[1]
    return (sender_layer[:, None] - base_layer[None, :]).abs() <= (width - 1) // 2


def gathered_mean(matrix: torch.Tensor, assignment: torch.Tensor) -> float:
    return float(matrix.gather(0, assignment[None, :]).mean())


def unflatten_assignment(assignment: torch.Tensor, sender_heads: int, receiver_shape) -> list:
    receiver_layers, receiver_heads, _ = receiver_shape
    grid = assignment.reshape(receiver_layers, receiver_heads)
    return [[[int(v) // sender_heads, int(v) % sender_heads] for v in row] for row in grid]


def save_plots(output_dir: Path, rel_k, rel_v, assignment, base, sender_shape, receiver_shape, tag: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sender_layers, sender_heads, _ = sender_shape
    receiver_layers, receiver_heads, _ = receiver_shape
    combined = 0.5 * (rel_k + rel_v)
    block = combined.reshape(sender_layers, sender_heads, receiver_layers, receiver_heads)

    layer_affinity = block.amin(dim=(1, 3))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(layer_affinity.numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xlabel("receiver layer")
    ax.set_ylabel("sender layer")
    ax.set_title(f"best-head relative MSE per layer pair ({tag}; lower=better)")
    layer_map = [int(b) // sender_heads for b in base.reshape(receiver_layers, receiver_heads)[:, 0]]
    ax.plot(range(receiver_layers), layer_map, "r--", linewidth=1, label="current proportional map")
    ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"layer_affinity_{tag}.png", dpi=150)
    plt.close(fig)

    chosen_layers = (assignment // sender_heads).reshape(receiver_layers, receiver_heads)
    fig, ax = plt.subplots(figsize=(9, 5))
    for head in range(receiver_heads):
        ax.scatter(range(receiver_layers), chosen_layers[:, head].numpy(), s=12, alpha=0.6)
    ax.plot(range(receiver_layers), layer_map, "r--", linewidth=1.5, label="current proportional map")
    ax.set_xlabel("receiver layer")
    ax.set_ylabel("chosen sender layer")
    ax.set_title(f"learned sender layer per receiver unit ({tag}; one dot per kv-head)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"chosen_layers_{tag}.png", dpi=150)
    plt.close(fig)

    picks = [0, receiver_layers // 2, receiver_layers - 1]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4))
    for ax, receiver_layer in zip(axes, picks):
        sender_layer = layer_map[receiver_layer]
        im = ax.imshow(block[sender_layer, :, receiver_layer, :].numpy(), cmap="viridis_r")
        ax.set_title(f"s{sender_layer} -> r{receiver_layer}")
        ax.set_xlabel("receiver head")
        ax.set_ylabel("sender head")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"head-affinity (relative MSE, {tag}) at proportional layer pairs")
    fig.tight_layout()
    fig.savefig(output_dir / f"head_affinity_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for matrix, label in ((rel_k, "K"), (rel_v, "V")):
        base_rows = matrix.gather(0, base[None, :]).reshape(receiver_layers, receiver_heads).mean(dim=1)
        learned_rows = matrix.gather(0, assignment[None, :]).reshape(receiver_layers, receiver_heads).mean(dim=1)
        (line,) = ax.plot(base_rows.numpy(), "--", label=f"{label} proportional")
        ax.plot(learned_rows.numpy(), "-", color=line.get_color(), label=f"{label} learned")
    ax.set_xlabel("receiver layer")
    ax.set_ylabel("held-out relative MSE")
    ax.set_title(f"proportional vs learned pairing ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"pairing_gain_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    features_path = Path(args.features)
    output_dir = Path(args.output_dir) if args.output_dir else features_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(features_path, map_location="cpu")
    meta = data["meta"]
    sender_shape = tuple(meta["sender_shape"])
    receiver_shape = tuple(meta["receiver_shape"])
    doc_index = data["doc_index"].to(torch.long)
    documents = doc_index.max().item() + 1
    print(f"features: {data['sender_key'].shape[0]} pairs, {documents} documents, "
          f"sender {sender_shape}, receiver {receiver_shape}")

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(documents, generator=generator)
    heldout_docs = order[: max(1, int(round(documents * args.heldout_fraction)))]
    heldout_mask = torch.isin(doc_index, heldout_docs)
    train_mask = ~heldout_mask
    print(f"split: {int(train_mask.sum())} train / {int(heldout_mask.sum())} held-out pairs "
          f"({documents - len(heldout_docs)}/{len(heldout_docs)} documents)")

    scores = {}
    for kind in ("key", "value"):
        x = flat_units(data[f"sender_{kind}"], args.device)
        y = flat_units(data[f"receiver_{kind}"], args.device)
        if args.rms_scales:
            x_scale = x[train_mask].pow(2).mean(dim=(0, 2)).sqrt().clamp_min(1e-3)
            y_scale = y[train_mask].pow(2).mean(dim=(0, 2)).sqrt().clamp_min(1e-3)
            x = x / x_scale[None, :, None]
            y = y / y_scale[None, :, None]
        rel_mse, cka = ridge_scores(
            x[train_mask], y[train_mask], x[heldout_mask], y[heldout_mask], args.ridge_lambda, args.receiver_chunk
        )
        y_ho = y[heldout_mask]
        mean_pred = y[train_mask].mean(dim=0)
        floor = (y_ho - mean_pred[None]).pow(2).sum(dim=(0, 2)) / y_ho.pow(2).sum(dim=(0, 2)).clamp_min(1e-12)
        scores[kind] = {"rel_mse": rel_mse, "cka": cka, "floor": floor.cpu()}
        del x, y, y_ho
        print(f"{kind}: scored {rel_mse.shape[0]}x{rel_mse.shape[1]} pairs "
              f"(best {float(rel_mse.min()):.3f}, median {float(rel_mse.median()):.3f}, "
              f"mean-predictor floor {float(floor.mean()):.3f})")

    rel_k, rel_v = scores["key"]["rel_mse"], scores["value"]["rel_mse"]
    base = baseline_assignment(sender_shape, receiver_shape)
    allowed = allowed_mask(sender_shape, receiver_shape, base, args.allowed_width)
    rel_k_masked = rel_k.where(allowed, torch.inf)
    rel_v_masked = rel_v.where(allowed, torch.inf)
    combined = 0.5 * (rel_k_masked + rel_v_masked)
    if args.tie_kv:
        key_assignment = value_assignment = combined.argmin(dim=0)
    else:
        key_assignment = rel_k_masked.argmin(dim=0)
        value_assignment = rel_v_masked.argmin(dim=0)

    summary = {
        "tie_kv": args.tie_kv,
        "rms_scales": args.rms_scales,
        "allowed_width": args.allowed_width,
        "ridge_lambda": args.ridge_lambda,
        "heldout_fraction": args.heldout_fraction,
        "pairs": int(data["sender_key"].shape[0]),
        "documents": documents,
        "floor": {"key": float(scores["key"]["floor"].mean()), "value": float(scores["value"]["floor"].mean())},
        "proportional": {"key": gathered_mean(rel_k, base), "value": gathered_mean(rel_v, base)},
        "learned": {"key": gathered_mean(rel_k, key_assignment), "value": gathered_mean(rel_v, value_assignment)},
        "learned_untied": {
            "key": float(rel_k_masked.min(dim=0).values.mean()),
            "value": float(rel_v_masked.min(dim=0).values.mean()),
        },
        "matches_proportional": {
            "key": float((key_assignment == base).float().mean()),
            "value": float((value_assignment == base).float().mean()),
        },
    }
    print(json.dumps(summary, indent=2))

    sender_heads = sender_shape[1]
    pairing = {
        **{k: summary[k] for k in ("tie_kv", "rms_scales", "allowed_width", "ridge_lambda")},
        "sender_model": meta["sender_model"],
        "receiver_model": meta["receiver_model"],
        "key_source": unflatten_assignment(key_assignment, sender_heads, receiver_shape),
        "value_source": unflatten_assignment(value_assignment, sender_heads, receiver_shape),
        "summary": summary,
    }
    tag = (f"{'tied' if args.tie_kv else 'untied'}{'_rms' if args.rms_scales else ''}"
           f"{f'_w{args.allowed_width}' if args.allowed_width else ''}")
    (output_dir / f"pairing_{tag}.json").write_text(json.dumps(pairing, indent=2))
    (output_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {"scores": scores, "summary": summary, "sender_shape": sender_shape, "receiver_shape": receiver_shape},
        output_dir / f"scores_{tag}.pt",
    )
    save_plots(output_dir, rel_k, rel_v, key_assignment, base, sender_shape, receiver_shape, tag)
    print(f"wrote pairing_{tag}.json, summary_{tag}.json, scores_{tag}.pt and 4 PNGs to {output_dir}")


if __name__ == "__main__":
    main()
