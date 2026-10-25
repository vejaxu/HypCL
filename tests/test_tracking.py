import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from models.hypcl import Learner
from utils.config import Experiment
from utils.tracking import ExperimentTracker


def config():
    return Experiment("cifar100", 50, 50, 2, 0.0002, 3, 0.7, "base")


class TrackingTests(unittest.TestCase):
    def test_epochs_and_task_evaluation_keep_distinct_axes(self):
        run = Mock()
        run.summary = {}
        with tempfile.TemporaryDirectory() as directory:
            with patch("wandb.init", return_value=run) as init:
                tracker = ExperimentTracker(config(), Path(directory))
        for task in (1, 2):
            for epoch in (1, 2):
                tracker.log_epoch(task=task, epoch=epoch, loss=1 / epoch,
                                  accuracy=80, learning_rate=0.001, seen_classes=task * 50)
            tracker.log_task(task=task, seen_classes=task * 50, average=78,
                             metrics={"top1": 75, "top5": 90,
                                      "grouped": {"old": None if task == 1 else 70,
                                                  "new": 80}})
        records = [call.args[0] for call in run.log.call_args_list]
        epochs = [row for row in records if "global_epoch" in row]
        evaluations = [row for row in records if "eval/accuracy" in row]
        self.assertEqual([row["global_epoch"] for row in epochs], [1, 2, 3, 4])
        self.assertEqual([row["task"] for row in evaluations], [1, 2])
        self.assertNotIn("eval/old_accuracy", evaluations[0])
        self.assertEqual(evaluations[1]["eval/old_accuracy"], 70)
        run.define_metric.assert_any_call("task_02/*", step_metric="task_02/epoch")
        self.assertEqual(init.call_args.kwargs["project"], "hypcli")
        self.assertNotIn("api_key", init.call_args.kwargs["config"])

    def test_training_loop_emits_real_losses_without_extra_evaluation(self):
        class TinyNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2, bias=False)

            def text_features(self, names, templates):
                return torch.eye(2)

            def encode_image(self, images):
                return self.layer(images)

        learner = object.__new__(Learner)
        learner.config = config()
        learner._device = torch.device("cpu")
        learner._network = TinyNetwork()
        learner._known_classes, learner._total_classes, learner._cur_task = 0, 2, 0
        learner.data_manager = SimpleNamespace(_class_to_label=["a", "b"],
                                               _data_to_prompt=["{}"])
        learner.metric_logger = Mock()
        learner._compute_accuracy = Mock(return_value=50)
        optimizer = torch.optim.SGD(learner._network.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        batch = (None, torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
        before = learner._network.layer.weight.detach().clone()
        learner.train_function([batch], [], optimizer, scheduler)
        self.assertEqual(learner._compute_accuracy.call_count, 2)
        rows = [call.kwargs for call in learner.metric_logger.call_args_list]
        self.assertEqual([row["epoch"] for row in rows], [1, 2])
        self.assertTrue(all(math.isfinite(row["loss"]) for row in rows))
        self.assertEqual(rows[0]["learning_rate"], 0.01)
        self.assertFalse(torch.equal(before, learner._network.layer.weight))

    def test_failure_finalizes_run_as_failed(self):
        import trainer

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(trainer, "ExperimentTracker") as tracker_type, \
                 patch.object(trainer, "_train", side_effect=RuntimeError("training failed")), \
                 patch.object(trainer.logging, "exception"):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    trainer.train(config(), directory, Path(directory) / "run", device="cpu")
                tracker_type.return_value.finish.assert_called_once_with(exit_code=1)
                tracker_type.return_value.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
