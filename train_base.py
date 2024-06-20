import torch 
import random
import math 
import pandas as pd 
from spookynet import SpookyNet
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
    "Cl": 17
} 

class SpookyBatch:
    device = torch.device('cuda')

    def __init__(self):
        self.N = 0
        self.Z = []
        self.R = []
        self.E = []
        self.F = []
        self.idx_i = []
        self.idx_j = []
        self.batch_seg = []

    def toTensor(self):
        self.Z = torch.tensor(self.Z,dtype=torch.int64,device=SpookyBatch.device)
        self.R = torch.tensor(self.R,dtype=torch.float32,device=SpookyBatch.device,requires_grad=True)
        self.Q = torch.zeros(self.N,dtype=torch.float32,device=SpookyBatch.device) # not using this so could just pass the same tensor around
        self.S = torch.zeros(self.N,dtype=torch.float32,device=SpookyBatch.device) # ditto
        self.E = torch.tensor(self.E,dtype=torch.float32,device=SpookyBatch.device)
        self.F = torch.tensor(self.F, dtype=torch.float32,device=SpookyBatch.device)
        self.idx_i = torch.tensor(self.idx_i,dtype=torch.int64,device=SpookyBatch.device)
        self.idx_j = torch.tensor(self.idx_j,dtype=torch.int64,device=SpookyBatch.device)
        self.batch_seg = torch.tensor(self.batch_seg,dtype=torch.int64,device=SpookyBatch.device) # int64 required for "index tensors"
        return self
    
class SpookyDataset:
    device = torch.device('cuda')

    def __init__(self):
        self.N = 0
        self.Z = []
        self.R = []
        self.E = []
        self.F = []
        self.idx_i = []
        self.idx_j = []
        self.batch_seg = []

    def toTensor(self):
        # TODO: remove device call with lightning later
        self.Z = torch.tensor(self.Z,dtype=torch.int64,device=SpookyBatch.device)
        self.R = torch.tensor(self.R,dtype=torch.float32,device=SpookyBatch.device,requires_grad=True)
        self.Q = torch.zeros(self.N,dtype=torch.float32,device=SpookyBatch.device) # not using this so could just pass the same tensor around
        self.S = torch.zeros(self.N,dtype=torch.float32,device=SpookyBatch.device) # ditto
        self.E = torch.tensor(self.E,dtype=torch.float32,device=SpookyBatch.device)
        #self.F = torch.tensor(self.F, dtype=torch.float32,device=SpookyBatch.device)
        self.idx_i = torch.tensor(self.idx_i,dtype=torch.int64,device=SpookyBatch.device)
        self.idx_j = torch.tensor(self.idx_j,dtype=torch.int64,device=SpookyBatch.device)
        self.batch_seg = torch.tensor(self.batch_seg,dtype=torch.int64,device=SpookyBatch.device) # int64 required for "index tensors"
        return self
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return (
            self.Z[idx],
            self.R[idx],
            self.E[idx],
            torch.tensor(self.F[idx], dtype=torch.float32,device=SpookyBatch.device),
            self.idx_i[idx],
            self.idx_j[idx],
            self.batch_seg[idx]
        )


class Dataloader(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
def load_batches(df): # my_mols == some structure which has your loaded mol data, prob retrieved from a file,
                                              # or you can load it from a file here on demand to save memory
    batches = []
    batch = None
    nm = 0 # how many mols in current batch
    NM = 100 # how many mols we put in each batch
    for ind, row in df.iterrows(): # assuming we have a pandas dataframe with the data
        if nm == 0:
            na = 0 # num total atoms in this batch
            batch = SpookyBatch() # stores the data in a format we can pass to SpookyNet

        pos_elem_list = [(elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]] 
        elem = [i[0] for i in pos_elem_list]
        pos = [i[1] for i in pos_elem_list]
        #print("elem list length: ", len(elem))
        #print("pos list length: ", len(pos))
        #print("force length: ", row["gradient"]) # errors cause this is jagged
        #print("energy: ", type(row["energy"]))
        force = row["gradient"]
        # flatten the force list
        force = [sublist for sublist in force]
        #print(force)
        energy = row["energy"]
        
        batch.Z.extend(elem)
        batch.R.extend(pos)
        batch.E.append(energy) # target energy
        batch.F.extend(force) # target forces
        cur_idx_i, cur_idx_j = get_idx(pos) # see below but also look at SpookyNetCalculator for more options
        cur_idx_i += na
        cur_idx_j += na
        batch.idx_i.extend(cur_idx_i)
        batch.idx_j.extend(cur_idx_j)
        batch.batch_seg.extend([nm]*len(elem))
        na += len(elem)
        nm += 1

        if nm >= NM:
            batch.N = nm
            batches.append(batch.toTensor()) # or you could convert to a tensor during training, depends on how much memory you have
            nm = 0 
    
    if batch:
        batches.append(batch.toTensor())

    return batches

def load_dataset(df): # my_mols == some structure which has your loaded mol data, prob retrieved from a file,
                                              # or you can load it from a file here on demand to save memory
    #batches = []
    dataset = SpookyDataset()
    nm = 0 # how many mols in current dataset
    na = 0 # num total atoms in this dataset
    
    for ind, row in df.iterrows(): # assuming we have a pandas dataframe with the data
        #if nm == 0:
        #    na = 0 # num total atoms in this batch
        #    batch = SpookyBatch() # stores the data in a format we can pass to SpookyNet

        pos_elem_list = [(elem_to_num[i["name"]], i["xyz"]) for i in row["molecule"]["sites"]] 
        elem = [i[0] for i in pos_elem_list]
        pos = [i[1] for i in pos_elem_list]
        #print("elem list length: ", len(elem))
        #print("pos list length: ", len(pos))
        #print("force length: ", row["gradient"]) # errors cause this is jagged
        #print("energy: ", type(row["energy"]))
        force = row["gradient"]
        # flatten the force list
        #force = [sublist for sublist in force]
        #print(force)
        energy = row["energy"]
        
        dataset.Z.extend(elem)
        dataset.R.extend(pos)
        dataset.E.append(energy) # target energy
        dataset.F.extend(force) # target forces
        cur_idx_i, cur_idx_j = get_idx(pos) # see below but also look at SpookyNetCalculator for more options
        cur_idx_i += na
        cur_idx_j += na
        dataset.idx_i.extend(cur_idx_i)
        dataset.idx_j.extend(cur_idx_j)
        dataset.batch_seg.extend([nm]*len(elem))
        na += len(elem)

    dataset.toTensor()

    return dataset

# taken from SpookyNetCalculator 
def get_idx(R):
    N = len(R)
    idx = torch.arange(N,dtype=torch.int64)
    idx_i = idx.view(-1, 1).expand(-1, N).reshape(-1)
    idx_j = idx.view(1, -1).expand(N, -1).reshape(-1)
    # exclude self-interactions
    nidx_i = idx_i[idx_i != idx_j]
    nidx_j = idx_j[idx_i != idx_j]
    #return nidx_i.numpy(),nidx_j.numpy() # kind of dumb converting to numpy when we use torch later, but it fits our model
    return nidx_i,nidx_j # kind of dumb converting to numpy when we use torch later, but it fits our model

def train():
    NUM_EPOCHES = 1000
    BEST_POINT = 'best.pt'
    START_LR = 1e-2

    model = SpookyNet().to(torch.float32).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(),lr=START_LR,amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold=0)

    df = pd.read_json("/home/santiagovargas/dev/berkeley_pes/data/test_libe.json")

    training = load_batches(df)
    validation = load_batches(df)
    training_dataset = load_dataset(df)
    mse_sum = torch.nn.MSELoss(reduction='sum')
    mse_sum_forces = torch.nn.MSELoss(reduction='sum')

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
            ) # energy only model tho 

            E = res_forces[0]
            F = res_forces[1]
            #print(F.shape)
            loss = (mse_sum(E, batch.E)+ mse_sum_forces(F, batch.F))/N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate = optimizer.param_groups[0]['lr']

        rmse, force_rmse = compute_rmse(validation,model) 
        
        rmse_sum = rmse + force_rmse
        if scheduler.is_better(rmse_sum, scheduler.best):
            model.save(BEST_POINT)
        scheduler.step(rmse_sum)
        if epoch % 10 == 0:
            print('Epoch: {} / LR: {} / RMSE: {:.3f} / F RMSE: {:.3f} / Best: {:.3f}'.format(scheduler.last_epoch, learning_rate, rmse, force_rmse, scheduler.best))

def compute_rmse(batches, model):
    mse_sum = torch.nn.MSELoss(reduction='sum')
    mse_sum_forces = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    total_forces_mse = 0.0
    count = 0
    model.eval()
    for batch in batches:
        N = batch.N
        #res = model.energy(Z=batch.Z,Q=batch.Q,S=batch.S,R=batch.R,idx_i=batch.idx_i,idx_j=batch.idx_j,batch_seg=batch.batch_seg,num_batch=N)
        res = model.energy_and_forces(
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
        E = res[0]
        F = res[1]

        # sum over pairings
        total_mse += mse_sum(E, batch.E).item()
        total_forces_mse += mse_sum_forces(F, batch.F).item()
        count += N

    model.train()
    return math.sqrt(total_mse / count), math.sqrt(total_forces_mse / count)

if __name__ == "__main__":
    train()