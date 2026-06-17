import lightning as L
import torch
import torch.optim as optim
from omegaconf import DictConfig

from simclr_fraud.models.encoder import TabularEncoder
from simclr_fraud.models.losses import nt_xent_loss, simclr_similarity_metrics
from simclr_fraud.models.projection import ProjectionHead


class LitSimCLR(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        model_cfg: DictConfig,
        pretrain_cfg: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "input_dim": input_dim,
                "lr": pretrain_cfg.lr,
                "weight_decay": pretrain_cfg.weight_decay,
                "temperature": pretrain_cfg.temperature,
            }
        )

        self.encoder = TabularEncoder(
            input_dim=input_dim,
            hidden_dim=model_cfg.hidden_dim,
            embedding_dim=model_cfg.embedding_dim,
            dropout=model_cfg.dropout,
        )
        self.projection_head = ProjectionHead(
            embedding_dim=model_cfg.embedding_dim,
            projection_dim=model_cfg.projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        x1, x2 = batch
        h1, z1 = self.project(x1)
        h2, z2 = self.project(x2)

        loss = nt_xent_loss(z1, z2, temperature=self.hparams.temperature)
        metrics = simclr_similarity_metrics(z1, z2)

        avg_embedding_norm = (h1.norm(dim=1).mean() + h2.norm(dim=1).mean()) / 2
        avg_projection_norm = (z1.norm(dim=1).mean() + z2.norm(dim=1).mean()) / 2
        embedding_std = torch.cat([h1, h2], dim=0).std(dim=0).mean()
        projection_std = torch.cat([z1, z2], dim=0).std(dim=0).mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            f"{stage}_positive_similarity",
            metrics["positive_similarity"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{stage}_negative_similarity",
            metrics["negative_similarity"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{stage}_similarity_gap",
            metrics["similarity_gap"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(f"{stage}_embedding_norm", avg_embedding_norm, on_step=True, on_epoch=True)
        self.log(f"{stage}_projection_norm", avg_projection_norm, on_step=True, on_epoch=True)
        self.log(f"{stage}_embedding_std", embedding_std, on_step=True, on_epoch=True)
        self.log(f"{stage}_projection_std", projection_std, on_step=True, on_epoch=True)

        if stage == "train" and self.trainer is not None and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(f"{stage}_lr", current_lr, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
