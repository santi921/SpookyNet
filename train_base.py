import torch
import random
import math
import pandas as pd
from spookynet import SpookyNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
elem_to_num = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
}


class SpookyBatch:
    device = torch.device("cuda")

    def __init__(self):
        self.N = 0
        self.Z = []
        self.R = []
        self.E = []
        self.F = []
        self.Q = []
        self.S = []
        self.idx_i = []
        self.idx_j = []
        self.batch_seg = []

    def toTensor(self):
        self.Z = torch.tensor(self.Z, dtype=torch.int64, device=SpookyBatch.device)
        self.R = torch.tensor(
            self.R, dtype=torch.float32, device=SpookyBatch.device, requires_grad=True
        )
        if self.Q == []:
            self.Q = torch.zeros(
                self.N, dtype=torch.float32, device=SpookyBatch.device
            )  # not using this so could just pass the same tensor around
        else:
            self.Q = torch.tensor(
                self.Q, dtype=torch.float32, device=SpookyBatch.device
            )
        if self.S == []:
            self.S = torch.zeros(
                self.N, dtype=torch.float32, device=SpookyBatch.device
            )  # ditto
        else:
            self.S = torch.tensor(
                self.S, dtype=torch.float32, device=SpookyBatch.device
            )
            
        self.E = torch.tensor(self.E, dtype=torch.float32, device=SpookyBatch.device)
        self.F = torch.tensor(self.F, dtype=torch.float32, device=SpookyBatch.device)
        self.idx_i = torch.tensor(
            self.idx_i, dtype=torch.int64, device=SpookyBatch.device
        )
        self.idx_j = torch.tensor(
            self.idx_j, dtype=torch.int64, device=SpookyBatch.device
        )
        self.batch_seg = torch.tensor(
            self.batch_seg, dtype=torch.int64, device=SpookyBatch.device
        )  # int64 required for "index tensors"
        return self


class SpookyDataset(Dataset):
    #device = torch.device("cuda")

    def __init__(self):
        self.molecules = []
        self.N = 0 

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        molecule = self.molecules[idx]
        return molecule
        

class SpookyDatasetTabular(Dataset):
    #device = torch.device("cuda")

    def __init__(self, df):
        self.df = df
        self.N = len(df)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        #molecule = self.molecules[idx]
        # grab row from dataframe
        row = self.df.iloc[idx]

        return row
        

class DataloaderMolecules(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

        def collate(samples):

            molecules = [sample for sample in samples]
            
            nm = 0 
            na = 0  
            batch = (
                SpookyBatch()
            ) 
                
            for molecule in molecules:
                
                elem = molecule["Z"]
                pos = molecule["R"]
                force = molecule["F"]
                force = [sublist for sublist in force]
                energy = molecule["E"]
                Q = molecule["Q"]
                S = molecule["S"]

                batch.Z.extend(elem)
                batch.R.extend(pos)
                batch.E.append(energy)  # target energy
                batch.F.extend(force)  # target forces
                batch.Q.extend([Q])
                batch.S.extend([S])
                cur_idx_i, cur_idx_j = get_idx(
                    pos
                ) 
                cur_idx_i += na
                cur_idx_j += na
                batch.idx_i.extend(cur_idx_i)
                batch.idx_j.extend(cur_idx_j)
                batch.batch_seg.extend([nm] * len(elem))
                na += len(elem)
                nm += 1
            
            batch.N = nm
            batch.toTensor()
            return batch

        super(DataloaderMolecules, self).__init__(dataset, collate_fn=collate, **kwargs)

class DataloaderTabular(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.df = dataset

        def collate(samples):

            # make dataframe from samples
            df_sub = pd.DataFrame(samples)
            
            nm = 0 
            na = 0  
            batch = (
                SpookyBatch()
            ) 
                
            for ind, row in df_sub.iterrows():
                pos_elem_list = [
                    (elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]
                ]

                elem = [i[0] for i in pos_elem_list]
                pos = [i[1] for i in pos_elem_list]
                force = row["gradient"]
                force = [sublist for sublist in force]
                energy = row["energy"]
                charge = row["molecule"]["charge"]
                spin = row["molecule"]["spin_multiplicity"]

                
                molecule_dict = {
                    "Z": elem,
                    "R": pos,
                    "E": energy,
                    "F": force,
                    "Q": charge,
                    "S": spin
                }


                batch.Z.extend(elem)
                batch.R.extend(pos)
                batch.E.append(energy)  # target energy
                batch.F.extend(force)  # target forces
                batch.Q.extend([charge])
                batch.S.extend([spin])
                cur_idx_i, cur_idx_j = get_idx(
                    pos
                ) 
                cur_idx_i += na
                cur_idx_j += na
                batch.idx_i.extend(cur_idx_i)
                batch.idx_j.extend(cur_idx_j)
                batch.batch_seg.extend([nm] * len(elem))
                na += len(elem)
                nm += 1
            
            batch.N = nm
            batch.toTensor()
            return batch

        super(DataloaderTabular, self).__init__(dataset, collate_fn=collate, **kwargs)  



def load_batches(
    df,
):  # my_mols == some structure which has your loaded mol data, prob retrieved from a file,
    # or you can load it from a file here on demand to save memory
    batches = []
    batch = None
    nm = 0  # how many mols in current batch
    NM = 100  # how many mols we put in each batch
    for ind, row in df.iterrows():  # assuming we have a pandas dataframe with the data
        if nm == 0:
            na = 0  # num total atoms in this batch
            batch = (
                SpookyBatch()
            )  # stores the data in a format we can pass to SpookyNet

        pos_elem_list = [
            (elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]
        ]
        elem = [i[0] for i in pos_elem_list]
        pos = [i[1] for i in pos_elem_list]
        force = row["gradient"]
        # flatten the force list
        force = [sublist for sublist in force]
        energy = row["energy"]

        batch.Z.extend(elem)
        batch.R.extend(pos)
        batch.E.append(energy)  # target energy
        batch.F.extend(force)  # target forces
        cur_idx_i, cur_idx_j = get_idx(
            pos
        )  # see below but also look at SpookyNetCalculator for more options

        cur_idx_i += na
        cur_idx_j += na
        batch.idx_i.extend(cur_idx_i)
        batch.idx_j.extend(cur_idx_j)
        batch.batch_seg.extend([nm] * len(elem))
        na += len(elem)
        nm += 1

        if nm >= NM:
            batch.N = nm
            batches.append(
                batch.toTensor()
            )  # or you could convert to a tensor during training, depends on how much memory you have
            nm = 0

    if batch:
        batches.append(batch.toTensor())

    return batches


def load_dataset(
    df,
):  # my_mols == some structure which has your loaded mol data, prob retrieved from a file,
    # or you can load it from a file here on demand to save memory
    # batches = []
    dataset = SpookyDataset()
    nm = 0  # how many mols in current dataset
    na = 0  # num total atoms in this dataset

    for ind, row in df.iterrows():  # assuming we have a pandas dataframe with the data
        pos_elem_list = [
            (elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]
        ]

        elem = [i[0] for i in pos_elem_list]
        pos = [i[1] for i in pos_elem_list]
        force = row["gradient"]
        energy = row["energy"]
        charge = row["molecule"]["charge"]
        spin = row["molecule"]["spin_multiplicity"]

        
        molecule_dict = {
            "Z": elem,
            "R": pos,
            "E": energy,
            "F": force,
            "Q": charge,
            "S": spin
        }
        
        dataset.molecules.append(molecule_dict)
        nm += 1

    dataset.N = nm

    return dataset


def get_idx(R):
    N = len(R)
    # gets all indices for pairs of atoms
    idx = torch.arange(N, dtype=torch.int64)
    
    # expand to all pairs
    idx_i = idx.view(-1, 1).expand(-1, N).reshape(-1)
    idx_j = idx.view(1, -1).expand(N, -1).reshape(-1)
    
    # exclude self-interactions
    nidx_i = idx_i[idx_i != idx_j]
    nidx_j = idx_j[idx_i != idx_j]
    # return nidx_i.numpy(),nidx_j.numpy() # kind of dumb converting to numpy when we use torch later, but it fits our model
    return (
        nidx_i,
        nidx_j,
    )  # kind of dumb converting to numpy when we use torch later, but it fits our model


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

        #random.shuffle(training)
        
        for batch in training_dataloader:
            
            N = batch.N
            #print(batch.idx_i)
            #print(batch.idx_j)
            #print(batch.N)
            #print(batch.Z)
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
                
            ) / N
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
    model.eval()
    for batch in batches:
        N = batch.N
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

    model.train()
    return math.sqrt(total_mse / count), math.sqrt(total_forces_mse / count)


def compute_rmse_dataloader(dataloader, model):
    mse_sum = torch.nn.MSELoss(reduction="sum")
    mse_sum_forces = torch.nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_forces_mse = 0.0
    count = 0
    model.eval()
    for batch in dataloader:
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
        dipole_pred = res_forces[5]

        # sum over pairings
        total_mse += mse_sum(batch.E, E_pred).item()
        total_forces_mse += mse_sum_forces(batch.F, F_pred).item()
        count += N

    model.train()
    return math.sqrt(total_mse / count), math.sqrt(total_forces_mse / count)


if __name__ == "__main__":
    #train()
    #train_new()
    train_tabular()
