import math
import torch
import numpy as np
import pandas as pd
from spookynet import SpookyNetLightning

from spookynet.data.dataset import SpookyDatasetTabular
from spookynet.data.dataloader import DataloaderTabular

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs


def eval_tabular():

    dipole = False
    batch_size = 128

    # load model here instead
    model = SpookyNetLightning(dipole=dipole).to(torch.float32).cuda()

    df = pd.read_json("./train_chunk_5_radqm9_20240807.json")
    df_val = pd.read_json("./train_chunk_5_radqm9_20240807.json")
    df_test = pd.read_json("./train_chunk_5_radqm9_20240807.json")

    dataset = SpookyDatasetTabular(df, dipole=dipole)
    dataset_val = SpookyDatasetTabular(df_val, dipole=dipole)
    dataset_test = SpookyDatasetTabular(df_test, dipole=dipole)

    training_dataloader = DataloaderTabular(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
    )
    validation_dataloader = DataloaderTabular(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    test_dataloader = DataloaderTabular(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    res_train = compute_rmse_dataloader(training_dataloader, model)
    res_val = compute_rmse_dataloader(validation_dataloader, model)
    res_test = compute_rmse_dataloader(test_dataloader, model)


def compute_rel_rmse(delta, target):
    target_norm = np.sqrt(np.mean(np.square(target))).item()
    return np.sqrt(np.mean(np.square(delta))) / (target_norm + 1e-9) * 100


def compute_rmse_dataloader(dataloader, model):

    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    mae_sum = torch.nn.L1Loss(reduction="sum")
    mae_sum_forces = torch.nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_forces_mse = 0.0
    total_mae = 0.0
    total_forces_mae = 0.0

    E_dev_list = []
    E_label_list = []
    F_dev_list = []
    F_label_list = []

    count = 0
    count_atoms = 0
    model.eval()

    for batch in dataloader:
        N = batch.N
        N_atoms = len(batch.Z)
        res_forces = model.energy_and_forces(
            Z=batch.Z,
            Q=batch.Q,
            S=batch.S,
            R=batch.R,
            idx_i=batch.idx_i,
            idx_j=batch.idx_j,
            batch_seg=batch.batch_seg,
            num_batch=N,
            create_graph=True
            # use_dipole=True
        )
        E_pred = res_forces[0]
        F_pred = res_forces[1]
        dipole_pred = res_forces[5]

        # sum over pairings
        total_mse += mse_sum(batch.E, E_pred).item()
        total_forces_mse += mse_sum_forces(batch.F, F_pred).item()
        total_mae += mae_sum(batch.E, E_pred).item()
        total_forces_mae += mae_sum_forces(batch.F, F_pred).item()

        # add to list for relative error
        E_dev_list.append(E_pred - batch.E)
        E_label_list.append(batch.E)
        F_dev_list.append(F_pred - batch.F)
        F_label_list.append(batch.F)

        count += N
        count_atoms += N_atoms

    model.train()

    ret_dict = {
        "E_rmse_per_atom": math.sqrt(total_mse / count_atoms),
        "F_rmse_per_atom": math.sqrt(total_forces_mse / count_atoms),
        "E_rmse": math.sqrt(total_mse / count),
        "F_rmse": math.sqrt(total_forces_mse / count),
        "E_mae": total_mae / count,
        "F_mae": total_forces_mae / count,
        "E_mae_per_atom": total_mae / count_atoms,
        "F_mae_per_atom": total_forces_mae / count_atoms,
        "E_mean_per_cent_absolute_error": compute_rel_rmse(
            torch.cat(E_dev_list), torch.cat(E_label_list)
        ),
        "F_mean_per_cent_absolute_error": compute_rel_rmse(
            torch.cat(F_dev_list), torch.cat(F_label_list)
        ),
    }

    return math.sqrt(total_mse / count_atoms), math.sqrt(total_forces_mse / count_atoms)


if __name__ == "__main__":

    eval_tabular()  # trains from df directly
