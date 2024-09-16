import torch
import random
import math
import pandas as pd

# from spookynet import SpookyNet
from spookynet import SpookyNetLightning

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from spookynet.data.dataset import SpookyDatasetTabular
from spookynet.data.dataloader import DataloaderTabular

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs


def train_tabular():

    dipole = False
    batch_size = 16

    model = SpookyNetLightning(
        dipole=dipole, 
        #zero_init=True,
    ).to(torch.float32).cuda()
    #loss_weights = {"E": 1, "F": 10, "D": 1}
    # optimizer = torch.optim.Adam(model.parameters(), lr=START_LR, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, factor=0.5, patience=50, threshold=0
    # )

    df = pd.read_json("./data/train_debug.json")
    df_val = pd.read_json("./data/train_debug.json")
    #df = pd.read_json("./data/train_chunk_5_radqm9_20240807.json")
    #df_val = pd.read_json("./data/train_chunk_5_radqm9_20240807.json")

    dataset = SpookyDatasetTabular(df, dipole=dipole)
    dataset_val = SpookyDatasetTabular(df_val, dipole=dipole)

    training_dataloader = DataloaderTabular(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    validation_dataloader = DataloaderTabular(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # logger_wb = WandbLogger(
    #    project="spooky_dev", name="test_logs_transfer"
    # )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./test/",
        filename="model_lightning_transfer_{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=True,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=1000,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        accumulate_grad_batches=1,
        strategy="auto",
        enable_progress_bar=True,
        gradient_clip_val=100.0,
        callbacks=[
            early_stopping_callback,
            lr_monitor,
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        default_root_dir="./test/",
        precision=32,
    )

    trainer.fit(model, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    train_tabular()  # trains from df directly
