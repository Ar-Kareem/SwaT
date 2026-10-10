from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from multiagent.model.unaware_comm import _base_model


class QueryToQueryTransfer(nn.Module):

    def __init__(
        self,
        sender_layers: int,
        sender_heads: int,
        sender_head_dim: int,
        receiver_layers: int | None = None,
        receiver_heads: int | None = None,
        receiver_head_dim: int | None = None,
        architecture: str = "linear",
        expansion: int = 8,
        pairing: list | None = None,
    ):
        super().__init__()
        self.sender_layers = int(sender_layers)
        self.sender_heads = int(sender_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.receiver_layers = int(receiver_layers if receiver_layers is not None else sender_layers)
        self.receiver_heads = int(receiver_heads if receiver_heads is not None else sender_heads)
        self.receiver_head_dim = int(
            receiver_head_dim if receiver_head_dim is not None else sender_head_dim
        )
        if architecture not in {"linear", "mlp"}:
            raise ValueError(f"unsupported query-transfer architecture {architecture!r}")
        if expansion <= 0:
            raise ValueError("query-transfer expansion must be positive")
        self.architecture = architecture
        self.expansion = int(expansion)
        self.num_layers = self.receiver_layers
        self.num_heads = self.receiver_heads
        self.head_dim = self.receiver_head_dim
        self.source_layers = self._make_index_map(self.sender_layers, self.receiver_layers)
        self.source_heads = self._make_index_map(self.sender_heads, self.receiver_heads)
        self.pairing = pairing
        if pairing is not None:
            if len(pairing) != self.receiver_layers or any(
                len(row) != self.receiver_heads for row in pairing
            ):
                raise ValueError(
                    f"pairing shape mismatch: expected {self.receiver_layers}x{self.receiver_heads}"
                )
            self.source_units = [[(int(sl), int(sh)) for sl, sh in row] for row in pairing]
            for row in self.source_units:
                for sl, sh in row:
                    if not (0 <= sl < self.sender_layers and 0 <= sh < self.sender_heads):
                        raise ValueError(f"pairing source ({sl}, {sh}) out of sender range")
        else:
            self.source_units = [
                [(self.source_layers[layer], self.source_heads[head]) for head in range(self.receiver_heads)]
                for layer in range(self.receiver_layers)
            ]
        self.transfer = nn.ModuleList([
            nn.ModuleList([
                self._build_projector()
                for _ in range(self.receiver_heads)
            ])
            for _ in range(self.receiver_layers)
        ])

    def _build_projector(self) -> nn.Module:
        if self.architecture == "mlp":
            hidden_dim = self.expansion * self.sender_head_dim
            return nn.Sequential(
                nn.Linear(self.sender_head_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.receiver_head_dim),
            )
        projector = nn.Linear(self.sender_head_dim, self.receiver_head_dim)
        with torch.no_grad():
            projector.weight.zero_()
            projector.bias.zero_()
            shared_dim = min(self.sender_head_dim, self.receiver_head_dim)
            projector.weight[:shared_dim, :shared_dim].copy_(torch.eye(shared_dim))
        return projector

    @staticmethod
    def _make_index_map(sender_count: int, receiver_count: int) -> list[int]:
        if receiver_count <= 1:
            return [0]
        sender_last = max(sender_count - 1, 0)
        receiver_last = receiver_count - 1
        return [round(index * sender_last / receiver_last) for index in range(receiver_count)]

    def forward_layer_all(self, receiver_layer: int, sender_queries: list[Tensor]) -> Tensor:
        if len(sender_queries) != self.sender_layers:
            raise ValueError(
                f"expected {self.sender_layers} sender layer tensors, got {len(sender_queries)}"
            )
        param = next(self.parameters())
        moved: dict[int, Tensor] = {}
        outputs = []
        for receiver_head, projector in enumerate(self.transfer[receiver_layer]):
            source_layer, source_head = self.source_units[receiver_layer][receiver_head]
            if source_layer not in moved:
                moved[source_layer] = sender_queries[source_layer].to(
                    device=param.device, dtype=param.dtype
                )
            outputs.append(projector(moved[source_layer][..., source_head, :]))
        return torch.stack(outputs, dim=-2)

    def forward_layer(self, receiver_layer: int, queries: Tensor) -> Tensor:
        if self.pairing is not None:
            raise ValueError(
                "forward_layer ignores per-unit pairings; use forward_layer_all"
            )
        if tuple(queries.shape[-2:]) != (self.sender_heads, self.sender_head_dim):
            raise ValueError(
                f"expected sender query tail {(self.sender_heads, self.sender_head_dim)}, "
                f"got {tuple(queries.shape[-2:])}"
            )
        selected = queries[..., self.source_heads, :]
        return torch.stack(
            [
                projector(selected[..., receiver_head, :])
                for receiver_head, projector in enumerate(self.transfer[receiver_layer])
            ],
            dim=-2,
        )


def load_query_transfer_state_dict(transfer: QueryToQueryTransfer, state_dict: dict) -> None:
    if "delta_weight" not in state_dict:
        transfer.load_state_dict(state_dict)
        return
    if transfer.architecture != "linear":
        raise ValueError("legacy query-transfer checkpoints can only be loaded as architecture='linear'")
    delta_weight = state_dict["delta_weight"]
    delta_bias = state_dict["delta_bias"]
    with torch.no_grad():
        for layer in range(transfer.receiver_layers):
            for head in range(transfer.receiver_heads):
                projector = transfer.transfer[layer][head]
                projector.weight.zero_()
                shared_dim = min(transfer.sender_head_dim, transfer.receiver_head_dim)
                projector.weight[:shared_dim, :shared_dim].copy_(torch.eye(shared_dim))
                projector.weight.add_(delta_weight[layer, head].T)
                projector.bias.copy_(delta_bias[layer, head])


def query_transfer_checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    direct = path / "query_transfer.pt"
    if direct.exists():
        return direct
    latest = path / "latest.json"
    if latest.exists():
        with open(latest) as stream:
            checkpoint = json.load(stream)["checkpoint"]
        candidate = path / checkpoint / "query_transfer.pt"
        if candidate.exists():
            return candidate
    candidates = list(path.glob("checkpoint-*/query_transfer.pt"))
    candidates.extend(path.glob("exit-checkpoint-*/query_transfer.pt"))
    if not candidates:
        raise FileNotFoundError(f"no query_transfer.pt checkpoint found under {path}")

    def step(candidate: Path) -> int:
        try:
            return int(candidate.parent.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=step)


def load_query_transfer(path: str | Path, device: str) -> tuple[Path, QueryToQueryTransfer]:
    checkpoint_path = query_transfer_checkpoint_file(path)
    state = torch.load(checkpoint_path, map_location="cpu")
    architecture = state.get("architecture", "linear")
    expansion = int(state.get("expansion", 8))
    if "sender_layers" in state:
        transfer = QueryToQueryTransfer(
            state["sender_layers"],
            state["sender_heads"],
            state["sender_head_dim"],
            state["receiver_layers"],
            state["receiver_heads"],
            state["receiver_head_dim"],
            architecture=architecture,
            expansion=expansion,
            pairing=state.get("pairing"),
        )
    else:
        transfer = QueryToQueryTransfer(
            state["num_layers"],
            state["num_heads"],
            state["head_dim"],
            architecture=architecture,
            expansion=expansion,
        )
    load_query_transfer_state_dict(transfer, state["query_transfer_state_dict"])
    transfer = transfer.to(device=device, dtype=torch.float32)
    return checkpoint_path, transfer.eval()


@torch.no_grad()
def collect_normalized_queries(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    query_positions: Tensor,
    *,
    past_key_values=None,
    output_device: str | torch.device = "cpu",
    output_dtype: torch.dtype | None = None,
) -> list[Tensor]:
    base = _base_model(model)
    captured: list[Tensor | None] = [None] * len(base.layers)
    handles = []
    for layer_idx, layer in enumerate(base.layers):
        attention = layer.self_attn

        def capture(_module, _inputs, output, *, index=layer_idx, attn=attention):
            selected = output[:, query_positions, :]
            selected = selected.view(
                selected.shape[0],
                selected.shape[1],
                -1,
                attn.head_dim,
            )
            q_norm = getattr(attn, "q_norm", None)
            if q_norm is not None:
                selected = q_norm(selected)
            captured[index] = selected.detach().to(device=output_device, dtype=output_dtype)

        handles.append(attention.q_proj.register_forward_hook(capture))
    try:
        output = base(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )
        del output
    finally:
        for handle in handles:
            handle.remove()
    if any(query is None for query in captured):
        raise RuntimeError("failed to capture sender queries from every layer")
    return [query for query in captured if query is not None]
