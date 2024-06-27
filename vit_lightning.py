import math
from functools import partial
import torch
import torch.nn as nn
import lightning as L
import torchmetrics.classification as cls_metrics

from vision_transformer import VisionTransformer


class ViTLightning(L.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        lr: float = 1e-4,
        opt: str = "adam",
        patience: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.block = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

        self.acc_metric = (
            cls_metrics.MulticlassAccuracy(
                num_classes=num_classes, average="micro"
            )
            if num_classes > 1
            else cls_metrics.BinaryAccuracy(multidim_average="global")
        )

    def forward(self, x, return_patch_embeddings: bool = False):
        return self.block(x, return_patch_embeddings=return_patch_embeddings)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)

        if self.block.num_classes == 1:
            pred = nn.functional.sigmoid(pred)
            loss = nn.functional.binary_cross_entropy(
                pred.squeeze(1), y.to(torch.float32)
            )
        else:
            loss = nn.functional.cross_entropy(pred, y)

        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        if self.hparams.opt == "adam":
            opt = torch.optim.AdamW(
                self.parameters(), self.hparams.lr, betas=(0.9, 0.999)
            )
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(
                self.parameters(), self.hparams.lr, momentum=0.9
            )

        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            factor=0.1,
            patience=self.hparams.patience,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)

        if self.block.num_classes == 1:
            pred = nn.functional.sigmoid(pred)
            loss = nn.functional.binary_cross_entropy(
                pred.squeeze(1), y.to(torch.float32)
            )

            acc = self.acc_metric(pred.squeeze(1), y)
        else:
            loss = nn.functional.cross_entropy(pred, y)
            acc = self.acc_metric(pred, y)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val_loss", loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)

        if self.block.num_classes == 1:
            pred = nn.functional.sigmoid(pred)
            loss = nn.functional.binary_cross_entropy(
                pred.squeeze(1), y.to(torch.float32)
            )

            acc = self.acc_metric(pred.squeeze(1), y)
        else:
            loss = nn.functional.cross_entropy(pred, y)
            acc = self.acc_metric(pred, y)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_acc",
            acc,
            on_epoch=True,
            sync_dist=True,
        )
        return {"test_loss", loss}


def vit_tiny(patch_size=16, **kwargs):
    model = ViTLightning(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = ViTLightning(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = ViTLightning(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
