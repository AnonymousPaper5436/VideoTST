#  import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
# from torch.utils.data.distributed import DistributedSampler

from . import _datamodules


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        # self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        for dm in self.dms:
            dm.setup(stage)

        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        self.tokenizer = self.dms[0].tokenizer

        self.collate = self.dms[0].train_dataset.collate

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            # drop_last=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
