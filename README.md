# dgraph_cogdl
cogdl version of Dgraph

This repo provides a collection of cogdl baselines for DGraphFin dataset. Please download the dataset from the DGraph web and place it under the folder 'dataset/'


## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch = 1.6.0  


## Training

- **MLP**
```bash
python gnn.py --model mlp --epochs 200 --runs 10 --device 0
```

- **GCN**
```bash
python gnn.py --model gcn --epochs 200 --runs 10 --device 0
```

- **GraphSAGE**
```bash
python gnn.py --model graphsage --epochs 200 --runs 10 --device 0
```

- **GAT**
```bash
python gnn.py --model gat --epochs 200 --runs 10 --device 0
```

- **Grand**
```bash
python gnn.py --model grand --epochs 200 --runs 10 --device 0
```

- **SGC**
```bash
python gnn.py --model sgc --epochs 200 --runs 10 --device 0
```

- **Mixhop**
```bash
python gnn.py --model mixhop --epochs 200 --runs 10 --device 0
```



## Results:
Performance on **DGraphFin**(10 runs):

| Methods   | Valid AUC  | Test AUC  |
|  :----  |  ---- | ---- |
| MLP | 0.7135 ± 0.0010 | 0.7192 ± 0.0009 |
| GCN | 0.7078 ± 0.0027 | 0.7078 ± 0.0023 |
| GraphSAGE| 0.7548 ± 0.0013 | 0.7621 ± 0.0017 |
| Grand  | 0.7674 ± 0.0005 | **0.7761 ± 0.0018** |
| GAT  | 0.7233 ± 0.0012 | 0.7333 ± 0.0024 |
| Mixhop | 0.7526 ± 0.0089 | 0.7624 ± 0.0081 |
| SGC | 0.7526 ± 0.0089 | 0.7624 ± 0.0081 |
