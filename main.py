import os
import vit_lightning as vit
from datamodule import ImageFolderModule
import argparse
import torch
from torchvision.transforms import v2 as T
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.loggers.csv_logs import CSVLogger


def main(args: argparse.Namespace):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    transforms = {
        "train": T.Compose(
            [
                T.RandomRotation(90),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.Resize((args.size, args.size)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((args.size, args.size)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        ),
    }
    datamodule = ImageFolderModule(
        args.root_dir,
        transforms=transforms["train"],
        test_dir=args.test_dir,
        test_transforms=transforms["test"],
        batch_size=args.batch_size,
    )

    assert args.num_classes >= 0, "Number of classes must not be negative"
    model = vit.__dict__[args.arch](
        patch_size=8,
        num_classes=args.num_classes,
        img_size=args.size,
        in_chans=args.in_chans,
        lr=args.lr,
        patience=args.patience // 2,
        opt="adam",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path, args.run),
        monitor="val_loss",
        save_last=True,
        save_top_k=3,
    )
    es_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    logger = CSVLogger(args.log_path)

    trainer = Trainer(
        default_root_dir=os.getcwd(),
        accelerator="gpu",
        devices=args.gpu_ids,
        strategy="ddp",
        check_val_every_n_epoch=1,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        log_every_n_steps=int(len(datamodule.dataset) * 0.8)
        // (datamodule.batch_size * len(args.gpu_ids)),
        callbacks=[
            checkpoint_callback,
            es_callback,
        ],
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", action="store", type=str, required=True)
    parser.add_argument(
        "--test_dir", action="store", type=str, required=False, default=None
    )
    parser.add_argument("--run", action="store", type=str, required=True)
    parser.add_argument(
        "--size", action="store", type=int, required=False, default=224
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        action="store",
        required=True,
    )
    parser.add_argument(
        "--in_chans", action="store", type=int, required=False, default=3
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, required=False, default=32
    )
    parser.add_argument(
        "--lr", action="store", type=float, required=False, default=1e-4
    )
    parser.add_argument(
        "--epochs", action="store", type=int, required=False, default=400
    )
    parser.add_argument(
        "--patience", action="store", type=int, required=False, default=20
    )
    parser.add_argument(
        "--arch",
        action="store",
        type=str,
        required=False,
        default="vit_base",
        choices=["vit_tiny", "vit_small", "vit_base"],
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        action="store",
        type=int,
        default=(0,),
        required=False,
    )
    parser.add_argument(
        "--checkpoint_path",
        action="store",
        type=str,
        required=False,
        default="checkpoints",
    )
    parser.add_argument(
        "--log_path",
        action="store",
        type=str,
        required=False,
        default=".",
    )

    args = parser.parse_args()
    main(args)
