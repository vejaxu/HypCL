"""W&B metrics for one class-incremental experiment (no model uploads)."""
from dataclasses import asdict
import logging

from utils.toolkit import BATCH_SIZE, CLASS_ORDER_SEED, PYTHON_SEED, TORCH_SEED


class ExperimentTracker:
    def __init__(self, config, output, *, project="hypcli", entity=None,
                 mode="online", name=None):
        self.run = None
        self.epochs = config.epochs
        self._tasks = set()
        if mode == "disabled":
            return
        import wandb

        initial = 0 if config.initial_classes == config.increment else config.initial_classes
        protocol = f"B{initial}_Inc{config.increment}"
        self.run = wandb.init(
            project=project, entity=entity, mode=mode, dir=str(output),
            name=name or f"{config.dataset}_{protocol}_{output.name}",
            group=config.dataset, job_type="train", tags=[config.dataset, protocol],
            config={**asdict(config), "protocol": protocol,
                    "batch_size": BATCH_SIZE, "optimizer": "AdamW",
                    "weight_decay": 0, "scheduler": "CosineAnnealingLR",
                    "backbone": "ViT-B-16", "pretrained": "laion400m_e32",
                    "class_order_seed": CLASS_ORDER_SEED,
                    "torch_seed": TORCH_SEED, "python_seed": PYTHON_SEED},
            save_code=False, settings=wandb.Settings(disable_code=True),
        )
        self.run.define_metric("train/*", step_metric="global_epoch")
        self.run.define_metric("eval/*", step_metric="task")
        self.run.summary["complete"] = False
        if mode == "online" and self.run.url:
            logging.info("W&B run: %s", self.run.url)

    def log_epoch(self, *, task, epoch, loss, accuracy, learning_rate, seen_classes):
        if self.run is None:
            return
        prefix = f"task_{task:02d}"
        if task not in self._tasks:
            self.run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")
            self._tasks.add(task)
        self.run.log({
            "global_epoch": (task - 1) * self.epochs + epoch,
            "train/task": task, "train/epoch": epoch,
            "train/loss": loss, "train/accuracy_uncalibrated": accuracy,
            "train/learning_rate": learning_rate, "train/seen_classes": seen_classes,
            f"{prefix}/epoch": epoch, f"{prefix}/loss": loss,
            f"{prefix}/accuracy_uncalibrated": accuracy,
        })

    def log_task(self, *, task, seen_classes, metrics, average):
        if self.run is None:
            return
        values = {"task": task, "eval/seen_classes": seen_classes,
                  "eval/accuracy": metrics["top1"], "eval/top5": metrics["top5"],
                  "eval/average_accuracy": average}
        for group in ("old", "new"):
            value = metrics["grouped"][group]
            if value is not None:
                values[f"eval/{group}_accuracy"] = value
        self.run.log(values)

    def complete(self, result):
        if self.run is not None:
            self.run.summary.update({
                "complete": result["complete"],
                "average_accuracy": result["average"],
                "final_accuracy": result["final"],
                "accuracy_curve": result["curve"],
                "elapsed_seconds": result["elapsed_seconds"],
            })

    def finish(self, exit_code=0):
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception:
                logging.exception("W&B finalization failed")
