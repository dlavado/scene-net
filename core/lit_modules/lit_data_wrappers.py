

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from core.datasets.ts40k import build_data_samples, TS40K


class LitTS40K(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for TS40K dataset.

    Parameters
    ----------

    `data_dir` - str :
        directory where the dataset is stored

    `batch_size` - int :
        batch size to use for routines

    `transform` - (None, torch.Transform) :
        transformation to apply to the point clouds

    `num_workers` - int :
        number of workers to use for data loading

    `val_split` - float :
        fraction of the training data to use for validation

    `test_split` - float :
        In the case of building the dataset from raw data, fraction of the data to use for testing

    """

    def __init__(self, data_dir, batch_size, transform=None, num_workers=8, val_split=0.1, test_split=0.4):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.save_hyperparameters()

    def _build_dataset(self, raw_data_dir): #Somewhat equivalent to `prepare_data` hook of LitDataModule
        build_data_samples(raw_data_dir, self.data_dir, tower_radius=True, data_split={"fit": 1 - self.hparams.test_split, 
                                                                                      "test": self.hparams.test_split})

    def setup(self, stage:str=None):
        # build dataset
        if stage == 'fit':
            fit_ds = TS40K(self.data_dir, split="fit", transform=self.transform)
            self.train_ds, self.val_ds = random_split(fit_ds, 
                                                      [len(fit_ds) - int(len(fit_ds) * self.hparams.val_split), int(len(fit_ds) * self.hparams.val_split)])
            del fit_ds
        if stage == 'test':
            self.test_ds = TS40K(self.data_dir, split="test", transform=self.transform)

        if stage == 'predict':
            self.predict_ds = TS40K(self.data_dir, split="test", transform=self.transform)

        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    # def state_dict(self):
    #     # track whatever you want here
    #     return 

    # def load_state_dict(self, state_dict):
    #     # restore the state based on what you tracked in (def state_dict)
    #     self.current_train_batch_index = state_dict["current_train_batch_index"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TS40K")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--val_split", type=float, default=0.1)
        parser.add_argument("--test_split", type=float, default=0.4)
        return parent_parser