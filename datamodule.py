import os
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

num_workers = os.cpu_count() // 4


class ImageFolderModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        transforms=None,
        test_dir: str = None,
        test_transforms=None,
        batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size

        self.root_dir = root_dir
        self.transforms = transforms

        self.test_dir = test_dir
        self.test_transforms = test_transforms

    def prepare_data(self):
        dataset = ImageFolder(self.root_dir, transform=self.transforms)
        self.train_set, self.val_set = random_split(
            dataset,
            (0.8, 0.2),
        )

        if self.test_dir:
            self.test_set = ImageFolder(self.test_dir, self.test_transforms)
        else:
            self.test_set = self.val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
