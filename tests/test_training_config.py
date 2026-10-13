import inspect
import unittest

from MG2Act.model import MG2ActModel
from MG2Act.train import build_parser


class TrainingConfigTests(unittest.TestCase):
    def test_defaults_match_documented_training_command(self):
        parser = build_parser()
        defaults = parser.parse_args(["--folder", "MG_data"])
        explicit = parser.parse_args([
            "--folder", "MG_data", "--epochs", "200", "--batch_size", "8",
            "--lr", "1e-4", "--embed_dim", "64", "--attn_heads", "4",
            "--decoder_layers", "2", "--mlp_hidden", "128,64", "--dropout", "0.1",
            "--proj_method", "conv", "--gnn_type", "gat", "--gnn_layers", "3",
            "--gnn_hidden_dim", "128", "--use_cosine", "--early_stop_patience", "30",
            "--fusion_method", "attention", "--seed", "42",
        ])
        self.assertEqual(vars(defaults), vars(explicit))
        self.assertEqual(inspect.signature(MG2ActModel).parameters["gnn_type"].default, "gat")

    def test_can_disable_scheduler_and_early_stopping(self):
        args = build_parser().parse_args([
            "--folder", "MG_data", "--no_cosine", "--early_stop_patience", "0",
        ])
        self.assertFalse(args.use_cosine)
        self.assertEqual(args.early_stop_patience, 0)


if __name__ == "__main__":
    unittest.main()
