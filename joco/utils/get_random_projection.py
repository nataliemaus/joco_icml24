import numpy as np 
from sklearn.random_projection import GaussianRandomProjection
import torch 

def get_random_projection(X, target_dim):
    transformer = GaussianRandomProjection(n_components=target_dim)
    X_new = transformer.fit_transform(X.detach().cpu().numpy())
    X_new = torch.from_numpy(X_new).float()
    return X_new

if __name__ == "__main__":
    X = torch.rand(25, 3000)
    print("X original shape", X.shape) # torch.Size([25, 3000])
    X_new = get_random_projection(X, target_dim=8)
    print("X new shape", X_new.shape) # torch.Size([25, 8])

