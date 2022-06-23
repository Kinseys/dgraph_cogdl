import torch
from cogdl import experiment
from cogdl.data import Graph
import numpy as np
import argparse

from cogdl.datasets import NodeDataset


def mask_change(id_mask,node_size):
    mask = torch.zeros(node_size).bool()
    for i in id_mask:
        mask[i]=True
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mlp,gcn,graphsage,grand,sgc,gat,mixhop')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=5e-7)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--runs', type=int, default=2)
    args = parser.parse_args()
    print(args)

    # Load data
    print('read_dgraphfin')
    folder = 'data/dgraphfin.npz'

    items = [np.load(folder)]

    #Create cogdl graph
    x = items[0]['x']
    y = items[0]['y'].reshape(-1, 1)

    edge_index = items[0]['edge_index']

    # set train/val/test mask in node_classification task
    train_id = items[0]['train_mask']
    valid_id = items[0]['valid_mask']
    test_id = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    #Feature normalization
    #x = (x - x.mean(0)) / x.std(0)

    y = torch.tensor(y, dtype=torch.int64)
    y = y.squeeze(1)

    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()

    # edge_type = torch.tensor(edge_type, dtype=torch.float)

    node_size = x.size()[0]

    train_m = torch.tensor(train_id, dtype=torch.int64)
    train_mask = mask_change(train_m, node_size)

    valid_m = torch.tensor(valid_id, dtype=torch.int64)
    valid_mask = mask_change(valid_m, node_size)

    test_m = torch.tensor(test_id, dtype=torch.int64)
    test_mask = mask_change(test_m, node_size)

    data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

    dataset = NodeDataset(data=data,scale_feat=False)
    #print(data)

    #Change runs to seed in cogdl
    seed = []
    for i in range(args.runs):
        seed.append(i)

    #Use cogdl experienment for training
    experiment(dataset=dataset, model=args.model,epochs=args.epochs,seed=seed ,hidden_size=args.hidden_size,lr=args.lr,l2=args.l2,early_stop=args.early_stop)

