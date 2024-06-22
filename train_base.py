import torch
import random
import math
import pandas as pd
from spookynet import SpookyNet
from spookynet.data.dataset import (
    SpookyBatch,
    SpookyDatasetTabular, 
    load_dataset, 
    load_batches
)

from spookynet.data.dataloader import (
    DataloaderMolecules,
    DataloaderTabular
)


def train_new():
    NUM_EPOCHES = 1000
    BEST_POINT = "best.pt"
    START_LR = 1e-2

    model = SpookyNet().to(torch.float32).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=50, threshold=0
    )

    df = pd.read_json("/home/santiagovargas/dev/berkeley_pes/data/test_libe.json")

    #training = load_batches(df)
    #validation = load_batches(df)
    training_dataset = load_dataset(df)

    batch_size = 100
    #print("training dataset length: ", len(training_dataset))
    training_dataloader = DataloaderMolecules(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = DataloaderMolecules(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    mse_sum_dipole = torch.nn.MSELoss(reduction="sum")

    for epoch in range(NUM_EPOCHES):

        #random.shuffle(training)
        
        for batch in training_dataloader:
            
            N = batch.N

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
            dipole = res_forces[2]
            
            loss = (
                mse_sum(batch.E, E_pred)
                + mse_sum_forces(batch.F, F_pred)
                
            ) / N
            #+ mse_sum_dipole(dipole, dipole_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate = optimizer.param_groups[0]["lr"]

        rmse, force_rmse = compute_rmse_dataloader(validation_dataloader, model)
        #rmse, force_rmse = compute_rmse(validation, model)
        rmse_sum = rmse + force_rmse
        
        if scheduler.is_better(rmse_sum, scheduler.best):
            model.save(BEST_POINT)
        scheduler.step(rmse_sum)
        if epoch % 10 == 0:
            print(
                "Epoch: {} / LR: {} / RMSE: {:.3f} / F RMSE: {:.3f} / Best: {:.3f}".format(
                    scheduler.last_epoch,
                    learning_rate,
                    rmse,
                    force_rmse,
                    scheduler.best,
                )
            )


def train_tabular():
    NUM_EPOCHES = 1000
    BEST_POINT = "best.pt"
    START_LR = 1e-2

    model = SpookyNet().to(torch.float32).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=50, threshold=0
    )

    df = pd.read_json("/home/santiagovargas/dev/berkeley_pes/data/test_libe.json")

    #training = load_batches(df)
    #validation = load_batches(df)
    #training_dataset = load_dataset(df)

    batch_size = 100
    #print("training dataset length: ", len(training_dataset))
    dataset = SpookyDatasetTabular(df)

    training_dataloader = DataloaderTabular(
        dataset, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = DataloaderTabular(
        dataset, batch_size=batch_size, shuffle=True
    )
    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    mse_sum_dipole = torch.nn.MSELoss(reduction="sum")

    for epoch in range(NUM_EPOCHES):

        for batch in training_dataloader:
            
            N = batch.N
            res_forces = model.forward(
                Z=batch.Z,
                Q=batch.Q,
                S=batch.S,
                R=batch.R,
                idx_i=batch.idx_i,
                idx_j=batch.idx_j,
                batch_seg=batch.batch_seg,
                num_batch=N,
                create_graph=True,
                use_forces=True, 
                use_dipole=True
            )

            E_pred = res_forces[0]
            F_pred = res_forces[1]
            dipole = res_forces[2]
            partial_charges = res_forces[5]
            
            
            loss = (
                mse_sum(batch.E, E_pred)
                + mse_sum_forces(batch.F, F_pred)
                
            ) / N
            #+ mse_sum_dipole(dipole, dipole_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate = optimizer.param_groups[0]["lr"]

        
        rmse, force_rmse = compute_rmse_dataloader(validation_dataloader, model)
        rmse_sum = rmse + force_rmse
        
        if scheduler.is_better(rmse_sum, scheduler.best):
            model.save(BEST_POINT)
        scheduler.step(rmse_sum)
        if epoch % 10 == 0:
            print(
                "Epoch: {} / LR: {} / E RMSE: {:.3f} / F RMSE: {:.3f} / Best: {:.3f}".format(
                    scheduler.last_epoch,
                    learning_rate,
                    rmse,
                    force_rmse,
                    scheduler.best,
                )
            )


def train():
    NUM_EPOCHES = 1000
    BEST_POINT = "best.pt"
    START_LR = 1e-3

    model = SpookyNet().to(torch.float32).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=50, threshold=0
    )

    df = pd.read_json("/home/santiagovargas/dev/berkeley_pes/data/test_libe.json")

    training = load_batches(df)
    validation = load_batches(df)
    #training_dataset = load_dataset(df)

    batch_size = 100
    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    mse_sum_dipole = torch.nn.MSELoss(reduction="sum")

    for epoch in range(NUM_EPOCHES):

        random.shuffle(training)
        for batch in training:
        
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
            )
            F = batch.F
            E = batch.E
            
            E_pred = res_forces[0]
            F_pred = res_forces[1]
            dipole = res_forces[2]
            
            # print(F.shape)
            loss = (
                mse_sum(E, E_pred)
                + mse_sum_forces(F, F_pred)
                
            ) / N_atoms
            #+ mse_sum_dipole(dipole, dipole_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate = optimizer.param_groups[0]["lr"]

        #rmse, force_rmse = compute_rmse_dataloader(validation_dataloader, model)
        rmse, force_rmse = compute_rmse(validation, model)
        rmse_sum = rmse + force_rmse
        
        if scheduler.is_better(rmse_sum, scheduler.best):
            model.save(BEST_POINT)
        scheduler.step(rmse_sum)

        if epoch % 10 == 0:
            print(
                "Epoch: {} / LR: {} / RMSE: {:.3f} / F RMSE: {:.3f} / Best: {:.3f}".format(
                    scheduler.last_epoch,
                    learning_rate,
                    rmse,
                    force_rmse,
                    scheduler.best,
                )
            )


def compute_rmse(batches, model):
    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_forces_mse = 0.0
    count = 0
    count_atoms = 0
    model.eval()
    for batch in batches:
        N = batch.N
        N_atoms = len(batch.Z)
        #print(batch.idx_i)
        #print(batch.idx_j)
        # res = model.energy(Z=batch.Z,Q=batch.Q,S=batch.S,R=batch.R,idx_i=batch.idx_i,idx_j=batch.idx_j,batch_seg=batch.batch_seg,num_batch=N)
        res = model.energy_and_forces(
            Z=batch.Z,
            Q=batch.Q,
            S=batch.S,
            R=batch.R,
            idx_i=batch.idx_i,
            idx_j=batch.idx_j,
            batch_seg=batch.batch_seg,
            num_batch=N,
            create_graph=True,
        )
        E = res[0]
        F = res[1]

        # sum over pairings
        total_mse += mse_sum(E, batch.E).item()
        total_forces_mse += mse_sum_forces(F, batch.F).item()
        count += N
        count_atoms += N_atoms

    model.train()
    return math.sqrt(total_mse / count_atoms), math.sqrt(total_forces_mse / count_atoms)


def compute_rmse_dataloader(dataloader, model):
    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_forces_mse = 0.0
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
        count += N
        count_atoms += N_atoms

    model.train()
    return math.sqrt(total_mse / count_atoms), math.sqrt(total_forces_mse / count_atoms)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    #train() # constructs all batches at once, memory intensive
    #train_tabular() # trains from df directly 
    
    train_new() # train from converted molecules dataset w/ batches
    