import sys
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.abl_closed_form.mapper import ClosedFormMapper, all_head_ols_r2, centered_ridge, ridge_from_moments, same_head_ols_r2
from multiagent.model.cache_renderer import flatten_sender_cache, pred_to_cache
from multiagent.model.unaware_comm import tuple_to_dynamic_cache


class ClosedFormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_ridge_against_augmented_least_squares(self):
        x = torch.randn(120, 7, dtype=torch.float64) + 3
        y = x @ torch.randn(7, 5, dtype=torch.float64) + torch.arange(5) + .05 * torch.randn(120, 5)
        ridge = .01
        w, b, residual = centered_ridge(x, y, ridge)
        design = torch.cat([x, torch.ones(120, 1)], dim=1)
        penalty = torch.cat([ridge ** .5 * torch.eye(7), torch.zeros(7, 1)], dim=1)
        expected = torch.linalg.lstsq(torch.cat([design, penalty]), torch.cat([y, torch.zeros(7, 5)])).solution
        torch.testing.assert_close(w, expected[:-1], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(b, expected[-1], rtol=1e-10, atol=1e-10)
        self.assertLess(residual, 1e-12)

    def test_chunked_moments_match_direct_ridge(self):
        x = torch.randn(300, 9, dtype=torch.float64) + 2
        y = x @ torch.randn(9, 4, dtype=torch.float64) + 1 + .1 * torch.randn(300, 4)
        w, b, _ = centered_ridge(x, y, .01)
        mx, my = x.mean(0), y.mean(0)
        gram = sum((x[a:a + 64] - mx).T @ (x[a:a + 64] - mx) for a in range(0, 300, 64))
        cross = sum((x[a:a + 64] - mx).T @ (y[a:a + 64] - my) for a in range(0, 300, 64))
        w2, b2, residual = ridge_from_moments(gram, cross, mx, my, .01)
        torch.testing.assert_close(w, w2, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(b, b2, rtol=1e-10, atol=1e-10)
        self.assertLess(residual, 1e-12)

    def test_rank_deficient_ols_and_head_order(self):
        x = torch.randn(2, 100, 4, dtype=torch.float64)
        x[:, :, 3] = x[:, :, 0]
        y = torch.cat([x @ torch.randn(2, 4, 4, dtype=torch.float64) + 2,
                       torch.randn(2, 100, 4, dtype=torch.float64)], dim=-1)
        scores = same_head_ols_r2(x, y)
        torch.testing.assert_close(scores[0], torch.ones(2, dtype=torch.float64))
        self.assertTrue((scores[1] < .2).all())

    def test_all_head_probe_matches_lstsq(self):
        x = torch.randn(200, 6, dtype=torch.float64) + 1
        y = x @ torch.randn(6, 10, dtype=torch.float64) + .5 * torch.randn(200, 10) + 3
        xc, yc = x - x.mean(0), y - y.mean(0)
        w = torch.linalg.lstsq(xc, yc).solution
        expected = 1 - (xc @ w - yc).square().sum() / yc.square().sum()
        self.assertAlmostEqual(all_head_ols_r2(x, y), float(expected), places=10)

    def test_selected_layer_all_head_affine_mapping(self):
        mapper = ClosedFormMapper(sender_layers=3, receiver_layers=2, kv_heads=2, head_dim=3, k=2, chunk_tokens=5)
        mapper.source_layers.copy_(torch.tensor([[2, 0], [1, 2]]))
        mapper.weight.normal_()
        mapper.bias.normal_()
        memory = torch.randn(1, 13, 3 * 2 * 6)
        positions = torch.arange(13)[None]
        result = mapper(None, positions, memory)
        source = memory.reshape(1, 13, 3, 2, 6)
        for target, selected in enumerate([[2, 0], [1, 2]]):
            for kind in range(2):
                x = torch.cat([source[:, :, s, kind] for s in selected], dim=-1)
                expected = x @ mapper.weight[target, kind] + mapper.bias[target, kind]
                torch.testing.assert_close(result[:, :, target, kind].flatten(-2), expected)

    def test_cross_tokenizer_gather_and_mismatched_heads(self):
        mapper = ClosedFormMapper(sender_layers=2, receiver_layers=3, kv_heads=3, head_dim=2,
                                  receiver_kv_heads=5, receiver_head_dim=4, k=1, chunk_tokens=4)
        mapper.source_layers.copy_(torch.tensor([[1], [0], [1]]))
        mapper.weight.normal_()
        mapper.bias.normal_()
        memory = torch.randn(1, 7, 2 * 2 * 6)
        limits = torch.tensor([[0, 1, 1, 2, 4, 4, 5, 7, 7]])
        result = mapper(None, torch.arange(9)[None], memory, memory_key_limits=limits)
        self.assertEqual(tuple(result.shape), (1, 9, 3, 2, 5, 4))
        index = (limits - 1).clamp(0, 6)[0]
        source = memory.reshape(1, 7, 2, 2, 6)[0, index]
        for target, s in enumerate([1, 0, 1]):
            for kind in range(2):
                expected = source[:, s, kind] @ mapper.weight[target, kind] + mapper.bias[target, kind]
                torch.testing.assert_close(result[0, :, target, kind].flatten(-2), expected)
        with self.assertRaises(ValueError):
            mapper(None, torch.arange(9)[None], memory)

    def test_identity_cache_matches_direct_qwen_decode(self):
        config = Qwen3Config(vocab_size=101, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
        model = Qwen3ForCausalLM(config).eval()
        self._check_identity_roundtrip(model, config, layers=2, heads=2, dim=8)

    def test_identity_cache_matches_direct_llama_mha_decode(self):
        config = LlamaConfig(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
        model = LlamaForCausalLM(config).eval()
        self._check_identity_roundtrip(model, config, layers=2, heads=4, dim=8)

    def _check_identity_roundtrip(self, model, config, layers, heads, dim):
        mapper = ClosedFormMapper(sender_layers=layers, receiver_layers=layers, kv_heads=heads, head_dim=dim)
        mapper.source_layers.copy_(torch.arange(layers)[:, None])
        for layer in range(layers):
            for kind in range(2):
                mapper.weight[layer, kind].copy_(torch.eye(heads * dim))
        ids = torch.randint(0, 101, (1, 18))
        positions = torch.arange(17)[None]
        limits = torch.arange(1, 18)[None]
        with torch.inference_mode():
            source = model(input_ids=ids[:, :-1], use_cache=True).past_key_values
            memory = flatten_sender_cache(model, source, positions)
            prediction = mapper(None, positions, memory, memory_key_limits=limits)
            transferred = pred_to_cache(model, prediction, positions, torch.float32)
            cache = tuple_to_dynamic_cache(transferred, config)
            actual = model(input_ids=ids[:, -1:], past_key_values=cache,
                           attention_mask=torch.ones_like(ids), position_ids=torch.tensor([[17]])).logits
            expected = model(input_ids=ids, use_cache=False).logits[:, -1:]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
