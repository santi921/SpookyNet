import torch 
from torch.utils.data import Dataset
from spookynet.data.dataloader import SpookyBatch
from spookynet.data.utils import elem_to_num, get_idx



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

    def __init__(self, df, dipole=False):
        self.df = df
        self.dipole = dipole
        self.N = len(df)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row
        


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
        #print(elem)

        dataset.molecules.append(molecule_dict)
        nm += 1
    print("... Done loading dataset")
    dataset.N = nm

    return dataset
