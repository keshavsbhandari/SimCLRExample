import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from simclr_fraud.models.classifier import FraudClassifier


class LitFraudClassifier(L.LightningModule):
    def __init__(
        self,
        model: FraudClassifier,
        pos_weight: torch.Tensor,
        finetune_cfg: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "lr": finetune_cfg.lr,
                "weight_decay": finetune_cfg.weight_decay,
            },
            ignore=["model", "pos_weight"],
        )

        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        batch_size = x.size(0)

        if stage == "train":
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            if self.trainer is not None and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log(
                    "train_lr",
                    current_lr,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
        else:
            self.log(
                f"{stage}_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        y_int = y.int()
        if stage == "val":
            self.val_auroc.update(probs, y_int)
            self.val_auprc.update(probs, y_int)
            self.val_precision.update(probs, y_int)
            self.val_recall.update(probs, y_int)
            self.val_f1.update(probs, y_int)
        elif stage == "test":
            self.test_auroc.update(probs, y_int)
            self.test_auprc.update(probs, y_int)
            self.test_precision.update(probs, y_int)
            self.test_recall.update(probs, y_int)
            self.test_f1.update(probs, y_int)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="test")

    def on_validation_epoch_end(self) -> None:
        self.log("val_roc_auc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_pr_auc", self.val_auprc.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)

        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self) -> None:
        self.log("test_roc_auc", self.test_auroc.compute())
        self.log("test_pr_auc", self.test_auprc.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())
        self.log("test_f1", self.test_f1.compute())

        self.test_auroc.reset()
        self.test_auprc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
