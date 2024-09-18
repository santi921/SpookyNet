import pandas as pd 
import torch
from torch.utils.data import DataLoader
from spookynet.data.utils import elem_to_num, get_idx
# import lightngin module 
from pytorch_lightning import LightningModule

class SpookyBatch:


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

    def toTensor(self, device=None):
        if device is not None:
            device = device
        else: 
            device = torch.device("cuda")

        self.Z = torch.tensor(
            self.Z, 
            dtype=torch.int64, 
            device=device
        )
        
        self.R = torch.tensor(
            self.R, 
            dtype=torch.float32, 
            device=device, 
            requires_grad=True
        )

        if self.Q == []:
            self.Q = torch.zeros(
                self.N, 
                dtype=torch.float32, 
                device=device
            )  # not using this so could just pass the same tensor around
        
        else:
            self.Q = torch.tensor(
                self.Q, 
                dtype=torch.float32, 
                device=device
            )

        if self.S == []:
            self.S = torch.zeros(
                self.N, 
                dtype=torch.float32, 
                device=device
            )  

        else:
            self.S = torch.tensor(
                self.S, 
                dtype=torch.float32, 
                device=device
            )
            
        self.E = torch.tensor(
                self.E, 
                dtype=torch.float32, 
                device=device
        )        
        
        self.F = torch.tensor(
                self.F, 
                dtype=torch.float32, 
                device=device
        )
        
        self.idx_i = torch.tensor(
            self.idx_i, 
            dtype=torch.int64, 
            device=device
        )

        self.idx_j = torch.tensor(
            self.idx_j, 
            dtype=torch.int64, 
            device=device
        )

        self.batch_seg = torch.tensor(
            self.batch_seg, 
            dtype=torch.int64, 
            device=device
        )  # int64 required for "index tensors"





        return self


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
    def __init__(self, dataset, device=torch.device("cuda"), **kwargs):
        self.df = dataset
        self.dipole = dataset.dipole
        # filter **kwargs for collate_fn
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}


        def collate_tabular(samples):

            # make dataframe from samples
            df_sub = pd.DataFrame(samples)
            
            nm = 0 
            na = 0

            Z_list = []
            R_list = []
            E_list = []
            F_list = []
            Q_list = []
            S_list = []
            dipole = []
            idx_i_list = []
            idx_j_list = []
            batch_seg_list = []
            
            #batch = (
            #    SpookyBatch()
            #) 
                
            for ind, row in df_sub.iterrows():

                pos_elem_list = [
                    (elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]
                ]
                charge = row["molecule"]["charge"]
                spin = row["molecule"]["spin_multiplicity"]

                elem = [i[0] for i in pos_elem_list]
                pos = [i[1] for i in pos_elem_list]
                force = row["gradient"]
                energy = row["relative_energy"]
                force = [sublist for sublist in force]
                dipole.append(row["resp_dipole_moments"])



                Z_list.extend(elem)
                R_list.extend(pos)
                E_list.append(energy) 
                F_list.extend(force)  
                Q_list.extend([charge])
                S_list.extend([spin])
                cur_idx_i, cur_idx_j = get_idx(
                    pos
                ) 
                cur_idx_i += na
                cur_idx_j += na
                idx_i_list.extend(cur_idx_i)
                idx_j_list.extend(cur_idx_j)
                batch_seg_list.extend([nm] * len(elem))
                
                na += len(elem)
                nm += 1
            
            N = nm
            device = torch.device("cuda")
            
            Z = torch.tensor(
                Z_list, 
                dtype=torch.int64, 
                device=device
            )

            R = torch.tensor(
                R_list, 
                dtype=torch.float32, 
                device=device, 
                requires_grad=True
            )

            if Q_list == []:
                Q = torch.zeros(
                    N, 
                    dtype=torch.float32, 
                    device=device
                )
            else:
                Q = torch.tensor(
                    Q_list, 
                    dtype=torch.float32, 
                    device=device
                )

            if S_list == []:
                S = torch.zeros(
                    N, 
                    dtype=torch.float32, 
                    device=device
                )
            else:
                S = torch.tensor(
                    S_list, 
                    dtype=torch.float32, 
                    device=device
                )

            E = torch.tensor(
                E_list, 
                dtype=torch.float32, 
                device=device, 
                requires_grad=True
            )

            F = torch.tensor(
                F_list, 
                dtype=torch.float32, 
                device=device
            )
            idx_i = torch.tensor(
                idx_i_list, 
                dtype=torch.int64, 
                device=device
            )

            idx_j = torch.tensor(
                idx_j_list, 
                dtype=torch.int64, 
                device=device
            )

            batch_seg = torch.tensor(
                batch_seg_list, 
                dtype=torch.int64, 
                device=device
            )  # int64 required for "index tensors"
            
            dipole = torch.tensor(
                dipole, 
                dtype=torch.float32, 
                device=device
            )
            #print('dipole dim:', dipole.shape)

            return {
                "Z": Z,
                "R": R,
                "E": E,
                "F": F,
                "Q": Q,
                "S": S,
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": batch_seg,
                "N": N,
                'D': dipole
            }



        def collate_tabular_no_dipole(samples):

            # make dataframe from samples
            df_sub = pd.DataFrame(samples)
            
            nm = 0 
            na = 0

            Z_list = []
            R_list = []
            E_list = []
            F_list = []
            Q_list = []
            S_list = []
            #dipole = []
            idx_i_list = []
            idx_j_list = []
            batch_seg_list = []
            
            #batch = (
            #    SpookyBatch()
            #) 
                
            for ind, row in df_sub.iterrows():

                pos_elem_list = [
                    (elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]
                ]
                charge = row["molecule"]["charge"]
                spin = row["molecule"]["spin_multiplicity"]

                elem = [i[0] for i in pos_elem_list]
                pos = [i[1] for i in pos_elem_list]
                force = row["gradient"]
                energy = row["relative_energy"]
                force = [sublist for sublist in force]
                #dipole.append(row["resp_dipole_moments"])



                Z_list.extend(elem)
                R_list.extend(pos)
                E_list.append(energy) 
                F_list.extend(force)  
                Q_list.extend([charge])
                S_list.extend([spin])
                cur_idx_i, cur_idx_j = get_idx(
                    pos
                ) 
                cur_idx_i += na
                cur_idx_j += na
                idx_i_list.extend(cur_idx_i)
                idx_j_list.extend(cur_idx_j)
                batch_seg_list.extend([nm] * len(elem))
                
                na += len(elem)
                nm += 1
            
            N = nm
            
            Z = torch.tensor(
                Z_list, 
                dtype=torch.int64, 
                device=device
            )

            R = torch.tensor(
                R_list, 
                dtype=torch.float32, 
                device=device, 
                requires_grad=True
            )

            if Q_list == []:
                Q = torch.zeros(
                    N, 
                    dtype=torch.float32, 
                    device=device
                )
            else:
                Q = torch.tensor(
                    Q_list, 
                    dtype=torch.float32, 
                    device=device
                )

            if S_list == []:
                S = torch.zeros(
                    N, 
                    dtype=torch.float32, 
                    device=device
                )
            else:
                S = torch.tensor(
                    S_list, 
                    dtype=torch.float32, 
                    device=device
                )

            E = torch.tensor(
                E_list, 
                dtype=torch.float32, 
                device=device, 
                requires_grad=True
            )

            F = torch.tensor(
                F_list, 
                dtype=torch.float32, 
                device=device
            )
            idx_i = torch.tensor(
                idx_i_list, 
                dtype=torch.int64, 
                device=device
            )

            idx_j = torch.tensor(
                idx_j_list, 
                dtype=torch.int64, 
                device=device
            )

            batch_seg = torch.tensor(
                batch_seg_list, 
                dtype=torch.int64, 
                device=device
            )  

            return {
                "Z": Z,
                "R": R,
                "E": E,
                "F": F,
                "Q": Q,
                "S": S,
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": batch_seg,
                "N": N
            }

            



        if self.dipole:
            super(DataloaderTabular, self).__init__(dataset, collate_fn=collate_tabular, **kwargs)
        else:
            super(DataloaderTabular, self).__init__(dataset, collate_fn=collate_tabular_no_dipole, **kwargs)  

class DataloaderMolecules(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.dipole = dataset.dipole
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super(DataloaderMolecules, self).__init__(dataset, collate_fn=collate_molecule, **kwargs)


