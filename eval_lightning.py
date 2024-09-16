import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from spookynet import SpookyNetLightning
from spookynet.data.dataset import SpookyDatasetTabular
from spookynet.data.dataloader import DataloaderTabular

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs


def eval_tabular():

    dipole = False
    batch_size = 128
    model_path = "./test/model_lightning_transfer_epoch=30-val_loss=2.3368.ckpt"

    # load model here instead
    #model = SpookyNetLightning(dipole=dipole).to(torch.float32).cuda()

    model = SpookyNetLightning.load_from_checkpoint(
        checkpoint_path=model_path
    ).to(torch.float32).cpu()
    
    # important!
    model.eval()
    
    print("... loaded model!")
    
    df = pd.read_json("./data/train_chunk_5_radqm9_20240807.json")
    df_val = pd.read_json("./data/train_chunk_5_radqm9_20240807.json")
    df_test = pd.read_json("./data/train_chunk_5_radqm9_20240807.json")

    dataset = SpookyDatasetTabular(df, dipole=dipole)
    dataset_val = SpookyDatasetTabular(df_val, dipole=dipole)
    dataset_test = SpookyDatasetTabular(df_test, dipole=dipole)

    training_dataloader = DataloaderTabular(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        device=torch.device("cpu")
    )
    validation_dataloader = DataloaderTabular(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        device=torch.device("cpu")
    )

    test_dataloader = DataloaderTabular(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        device=torch.device("cpu")
    )

    res_train = compute_rmse_dataloader(training_dataloader, model)
    print(res_train)
    
    res_val = compute_rmse_dataloader(validation_dataloader, model)
    print(res_val)
    
    res_test = compute_rmse_dataloader(test_dataloader, model)
    print(res_test)

def compute_rel_rmse(delta, target):
    target_norm = torch.sqrt(torch.mean(torch.square(target))).item()
    return torch.sqrt(torch.mean(torch.square(delta))) / (target_norm + 1e-9) * 100


def compute_rmse_dataloader(dataloader, model, dipole=False):

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

    for batch in tqdm(dataloader):
        
        N = batch["N"]
        N_atoms = len(batch["Z"])
        Z = batch["Z"]
        Q = batch["Q"]
        S = batch["S"]
        R = batch["R"]
        idx_i = batch["idx_i"]
        idx_j = batch["idx_j"]
        batch_seg = batch["batch_seg"]
        E = batch["E"]
        F = batch["F"]

        res_forces = model.energy_and_forces(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            batch_seg=batch_seg,
            num_batch=N,
            create_graph=True
            #use_dipole=dipole
        )

        
        # sum over pairings
        E_pred = res_forces[0]
        F_pred = res_forces[1]
        E_true = batch["E"]
        F_true = batch["F"]

        total_mse += mse_sum(E_true, E_pred).item()
        total_forces_mse += mse_sum_forces(F_true, F_pred).item()
        total_mae += mae_sum(E_true, E_pred).item()
        total_forces_mae += mae_sum_forces(F_true, F_pred).item()

        # add to list for relative error
        dev_E = E_pred - E_true
        dev_F = F_pred - F_true
        # move to numpy 
        dev_E = dev_E.detach().numpy()
        dev_F = dev_F.detach().numpy()

        E_dev_list.append(dev_E)
        E_label_list.append(E_true.detach().numpy())
        F_dev_list.append(dev_F)
        F_label_list.append(F_true.detach().numpy())

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
        ).detach().cpu().numpy(),
        "F_mean_per_cent_absolute_error": compute_rel_rmse(
            torch.cat(F_dev_list), torch.cat(F_label_list)
        ).detach().cpu().numpy(),
    }
    return ret_dict

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    eval_tabular()  # trains from df directly
