import copy
import tempfile
import unittest
from pathlib import Path

import torch

from test_sequence_cache import make_encoder


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_new_encoder_reuses_disk_and_trains_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_encoder(cache_dir=directory).eval()
            expected = first(["ACDE"]).detach()
            state = copy.deepcopy(first.state_dict())
            second = make_encoder(cache_dir=directory)
            second.load_state_dict(state)
            calls = second.esmc.calls
            actual = second(["ACDE"])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(second.esmc.calls, calls)
            self.assertEqual(second.get_cache_info()["disk_hits"], 1)
            actual.sum().backward()
            self.assertIsNotNone(second.protein_projector.weight.grad)
            self.assertTrue(all(p.grad is None for p in second.esmc.parameters()))

    def test_projection_changes_reuse_raw_cache_but_backbone_changes_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_encoder(cache_dir=directory).eval()
            reference = first(["ACDE"]).detach()
            state = copy.deepcopy(first.state_dict())
            state["protein_projector.bias"] += 1
            first.load_state_dict(state)
            calls = first.esmc.calls
            torch.testing.assert_close(first(["ACDE"]), reference + 1)
            self.assertEqual(first.esmc.calls, calls)
            state["esmc.embed.weight"] += 1
            first.load_state_dict(state)
            first(["ACDE"])
            self.assertEqual(first.esmc.calls, calls + 1)
            self.assertEqual(len(list(Path(directory).glob("*/*.pt"))), 2)

    def test_corrupt_and_wrong_sequence_cache_are_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(cache_dir=directory).eval()
            expected = encoder(["ACDE"]).detach()
            path = next(Path(directory).glob("*/*.pt"))
            path.write_bytes(b"interrupted or invalid cache")
            encoder.clear_cache()
            calls = encoder.esmc.calls
            with self.assertWarns(RuntimeWarning):
                torch.testing.assert_close(encoder(["ACDE"]), expected)
            self.assertEqual(encoder.esmc.calls, calls + 1)
            payload = torch.load(path, weights_only=True)
            payload["sequence"] = "DIFFERENT"
            torch.save(payload, path)
            encoder.clear_cache()
            with self.assertWarns(RuntimeWarning):
                torch.testing.assert_close(encoder(["ACDE"]), expected)

    def test_inference_created_disk_cache_can_train_and_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(cache_dir=directory).eval()
            with torch.inference_mode():
                encoder(["ACDE"])
            encoder.clear_cache()
            encoder.train()
            encoder(["ACDE"]).sum().backward()
            self.assertIsNotNone(encoder.protein_projector.weight.grad)
            self.assertEqual(encoder.get_cache_info()["disk_hits"], 1)
            disabled = make_encoder(cache_dir="off")
            disabled(["ACDE"])
            self.assertIsNone(disabled._disk_cache)


if __name__ == "__main__":
    unittest.main()
