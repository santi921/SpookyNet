import torch 

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

