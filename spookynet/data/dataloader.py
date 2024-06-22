import pandas as pd 
import torch
from torch.utils.data import DataLoader
from spookynet.data.utils import elem_to_num, get_idx

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


def collate_tabular(samples):

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


def collate_molecule(samples):

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
        cur_idx_i += batch.Z.extend(elem)
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


class DataloaderTabular(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.df = dataset
        super(DataloaderTabular, self).__init__(dataset, collate_fn=collate_tabular, **kwargs)  

class DataloaderMolecules(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        super(DataloaderMolecules, self).__init__(dataset, collate_fn=collate_molecule, **kwargs)


