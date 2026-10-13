import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from MG2Act.encoder import SequenceEncoder


class FakeTokenizer:
    def encode(self, sequence):
        return [0] + [ord(c) % 16 for c in sequence] + [1]

    def get_vocab(self):
        return {str(i): i for i in range(16)}


class FakeESMC(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.calls = 0

    def logits(self, protein, config):
        self.calls += 1
        return SimpleNamespace(embeddings=self.embed(protein.sequence).unsqueeze(0))


def make_encoder(method="linear", cache_dir=""):
    with patch("MG2Act.encoder.ESMC.from_pretrained", return_value=FakeESMC()), \
         patch("MG2Act.encoder.EsmSequenceTokenizer", FakeTokenizer):
        return SequenceEncoder(proj_method=method, embed_dim=4, dropout=0,
                               embedding_cache_dir=cache_dir)


class SequenceCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_repeated_sequences_keep_projector_trainable(self):
        for method in ("linear", "conv", "mlp"):
            for token_level in (False, True):
                with self.subTest(method=method, token_level=token_level):
                    encoder = make_encoder(method).train()
                    optimizer = torch.optim.SGD(encoder.protein_projector.parameters(), lr=0.02)
                    initial_calls = encoder.esmc.calls
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        before = [p.detach().clone() for p in encoder.protein_projector.parameters()]
                        result = encoder(["ACDE", "ACDE"], return_token_level=token_level)
                        output = torch.stack(result) if token_level else result
                        target = torch.arange(output.numel()).reshape_as(output) / output.numel()
                        (output - target).square().mean().backward()
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                            for p in encoder.protein_projector.parameters()))
                        self.assertTrue(all(p.grad is None for p in encoder.esmc.parameters()))
                        optimizer.step()
                        self.assertTrue(any(not torch.equal(a, b) for a, b in
                                            zip(before, encoder.protein_projector.parameters())))
                    self.assertEqual(encoder.esmc.calls - initial_calls, 1)
                    self.assertFalse(encoder.esmc.training)
                    self.assertEqual(encoder.cache["ACDE"].shape[-1], 8)

    def test_cached_and_fresh_outputs_and_gradients_match(self):
        for method in ("linear", "conv", "mlp"):
            for token_level in (False, True):
                encoder = make_encoder(method).eval()  # disable dropout, retain autograd
                encoder.preload_cache(["ACDE", "ACDE"])
                def run():
                    encoder.zero_grad(set_to_none=True)
                    result = encoder(["ACDE"], return_token_level=token_level)
                    result = result[0]
                    result.square().sum().backward()
                    return result.detach().clone(), [p.grad.clone() for p in encoder.protein_projector.parameters()]
                cached, cached_grad = run()
                encoder.clear_cache()
                fresh, fresh_grad = run()
                torch.testing.assert_close(cached, fresh)
                for a, b in zip(cached_grad, fresh_grad):
                    torch.testing.assert_close(a, b)

    def test_cache_uses_latest_projection_and_survives_inference_to_train(self):
        encoder = make_encoder().eval()
        with torch.inference_mode():
            before = encoder(["ACDE"])
        with torch.no_grad():
            encoder.protein_projector.bias.add_(1)
        encoder.train()
        after = encoder(["ACDE"])
        torch.testing.assert_close(after, before + 1)
        after.sum().backward()
        self.assertIsNotNone(encoder.protein_projector.weight.grad)

    def test_state_load_and_backbone_changes_invalidate_cache(self):
        encoder = make_encoder().eval()
        state = copy.deepcopy(encoder.state_dict())
        reference = encoder(["ACDE"])
        keys = set(encoder.state_dict())
        self.assertEqual(keys, set(state))
        encoder.load_state_dict(state, strict=True)
        self.assertFalse(encoder.cache)
        torch.testing.assert_close(encoder(["ACDE"]), reference)
        calls = encoder.esmc.calls
        with torch.no_grad():
            encoder.esmc.embed.weight.add_(1)
        changed = encoder(["ACDE"])
        self.assertEqual(encoder.esmc.calls, calls + 1)
        self.assertFalse(torch.equal(reference, changed))

    def test_unfrozen_backbone_bypasses_cache(self):
        encoder = make_encoder()
        encoder.preload_cache(["ACDE"])
        encoder.esmc.embed.weight.requires_grad_(True)
        encoder.train()
        calls = encoder.esmc.calls
        encoder(["ACDE", "ACDE"]).sum().backward()
        self.assertEqual(encoder.esmc.calls, calls + 2)
        self.assertFalse(encoder.cache)
        self.assertIsNotNone(encoder.esmc.embed.weight.grad)


if __name__ == "__main__":
    unittest.main()
