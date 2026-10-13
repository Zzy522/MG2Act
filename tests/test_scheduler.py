import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from MG2Act import train


class SchedulerTests(unittest.TestCase):
    def run_training(self, cosine=True, patience=0, losses=(2.0, 1.0, 3.0)):
        rates, optimizers, schedulers = [], [], []
        real_optimizer = torch.optim.AdamW
        real_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

        class TinyModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

            def forward(self, e3, target, molecule):
                rates.append(optimizers[0].param_groups[0]["lr"])
                return self.weight.expand(len(e3))

        def make_optimizer(*args, **kwargs):
            optimizer = real_optimizer(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        def make_scheduler(*args, **kwargs):
            scheduler = real_scheduler(*args, **kwargs)
            schedulers.append(scheduler)
            return scheduler

        samples = [{"e3_seq": "AC", "target_seq": "DE", "smiles": "CC",
                    "Score": torch.tensor([0.2])} for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            args = train.build_parser().parse_args([
                "--folder", directory, "--out", directory, "--epochs", "3",
                "--batch_size", "2", "--lr", "0.01", "--quiet",
                "--early_stop_patience", str(patience),
                "--use_cosine" if cosine else "--no_cosine",
            ])
            with patch.object(train, "MG2ActDataset", return_value=samples), \
                 patch.object(train, "MG2ActModel", TinyModel), \
                 patch.object(train, "evaluate", side_effect=[(v, v) for v in losses]) as evaluation, \
                 patch.object(train.torch.optim, "AdamW", side_effect=make_optimizer), \
                 patch.object(train.torch.optim.lr_scheduler, "CosineAnnealingLR", side_effect=make_scheduler), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = train.train_fixed_split("train.csv", "val.csv", args, torch.device("cpu"))
            self.assertTrue((Path(directory) / "mg2act_best.pt").is_file())
            self.assertEqual(evaluation.call_count, result[3])
        return rates, schedulers, result

    def test_epoch_scheduler_and_validation_selection(self):
        rates, schedulers, result = self.run_training()
        self.assertEqual(len(rates), 9)  # three batches per epoch
        for observed, expected in zip(rates, [0.01] * 3 + [0.0075] * 3 + [0.0025] * 3):
            self.assertAlmostEqual(observed, expected)
        self.assertEqual(schedulers[0].last_epoch, 3)
        self.assertEqual(result[4], 2)  # best validation loss is epoch 2

    def test_disabled_scheduler_keeps_constant_rate(self):
        rates, schedulers, _ = self.run_training(cosine=False)
        self.assertEqual(rates, [0.01] * 9)
        self.assertFalse(schedulers)

    def test_early_stopping_updates_once_per_completed_epoch(self):
        rates, schedulers, result = self.run_training(patience=1, losses=(1.0, 2.0, 3.0))
        self.assertEqual(len(rates), 6)
        self.assertEqual(result[3:5], (2, 1))
        self.assertEqual(schedulers[0].last_epoch, 2)


if __name__ == "__main__":
    unittest.main()
