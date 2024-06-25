import math
from functools import partial
import torch
import torch.nn as nn
import lightning as L

from vision_transformer import PatchEmbed, Block


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

        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.num_classes = num_classes
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Init model weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w: int, h: int):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size

        w0, h0 = (
            w0 + 0.1,
            h0 + 0.1,
        )  # add small number for avoiding floating point error in the interpolation
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, _, h, w = x.shape
        x = self.patch_embed(x)

        # add [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # add positional embedding
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_patch_embeddings: bool = False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # return full cls and patch tokens
        if return_patch_embeddings:
            return x

        # return classification output
        out = self.head(x[:, 0])
        return out

    def get_last_self_attention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return last block self attention mask
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def configure_optimizers(self):
        if self.hparams.opt == "adam":
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999)
            )
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
            )

        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            factor=0.1,
            patience=self.hparams.patience,
        )
        return [opt], [lr_sched]

    def training_step(self, batch, _):
        x, y = batch

        pred = self.forward(x)

        if self.num_classes == 1:
            pred = nn.functional.sigmoid(pred)
            loss = nn.functional.binary_cross_entropy(pred.squeeze(1), y)
        else:
            loss = nn.functional.cross_entropy(pred, y)

        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, _):
        x, y = batch

        pred = self.forward(x)

        if self.num_classes == 1:
            pred = nn.functional.sigmoid(pred)
            loss = nn.functional.binary_cross_entropy(pred.squeeze(1), y)
        else:
            loss = nn.functional.cross_entropy(pred, y)
        self.log("val_loss", loss, on_epoch=True)

        if self.num_classes == 1:
            pred = torch.round(pred)
            pred = pred.squeeze(1)
        else:
            pred = torch.argmax(pred, dim=1)
        acc = (y == pred).sum().item() / y.size(0)
        self.log("val_acc", acc, on_epoch=True)

        return {"val_loss": loss}

    def test_step(self, batch, _):
        x, y = batch

        pred = self.forward(x)

        if self.num_classes == 1:
            pred = nn.functional.sigmoid(pred)
            loss = nn.functional.binary_cross_entropy(pred.squeeze(1), y)
        else:
            loss = nn.functional.cross_entropy(pred, y)
        self.log("test_loss", loss, on_epoch=True)

        if self.num_classes == 1:
            pred = torch.round(pred)
            pred = pred.squeeze(1)
        else:
            pred = torch.argmax(pred, dim=1)
        acc = (y == pred).sum().item() / y.size(0)
        self.log("test_acc", acc, on_epoch=True)

        return {"test_loss": loss}


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
