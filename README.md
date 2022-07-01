# dgraph_cogdl
cogdl version of Dgraph

This repo provides a collection of cogdl baselines for DGraphFin dataset. Please download the dataset from the DGraph web and place & unzip it under the folder 'dataset/'

**Dgraph dataset:** https://dgraph.oss-cn-shanghai.aliyuncs.com/DGraphFin.zip

**Cogdl introduction:** https://cogdl.readthedocs.io/en/latest/index.html

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch >= 1.6.0  
- pillow = 9.1.1

## Training

- **MLP**
```bash
python gnn.py --model mlp --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GCN**
```bash
python gnn.py --model gcn --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GraphSAGE**
```bash
python gnn.py --model graphsage --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GIN**
```bash
python gnn.py --model gin --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GAT**
```bash
python gnn.py --model gat --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **Grand**
```bash
python gnn.py --model grand --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **SGC**
```bash
python gnn.py --model sgc --epochs 200 --runs 10 --device 0
```

- **Mixhop**
```bash
python gnn.py --model mixhop --epochs 200 --runs 10 --device 0
```

- **You can find more models usage of function experiment() on cogdl https://cogdl.readthedocs.io/en/latest/index.html**


## Results:
Performance on **DGraphFin**(10 runs):

| Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  |  ---- |  ---- | ---- |
| MLP | 0.7102 ± 0.0033 | 0.6987 ± 0.0029 | 0.7059 ± 0.0030 |
| GCN | 0.7187 ± 0.0039 | 0.7093 ± 0.0048 | 0.7115 ± 0.0025 |
| GraphSAGE| 0.7687 ± 0.0022 | 0.7521 ± 0.0021 | **0.7601 ± 0.0013** |
| GIN | 0.7774 ± 0.0075 | 0.7594 ± 0.0069 | 0.7676 ± 0.0062 |
| Grand  | 0.6987 ± 0.0029 | 0.6935 ± 0.0025 | 0.6955 ± 0.0020 |
| GAT  | 0.6987 ± 0.0029 | 0.7233 ± 0.0012 | 0.7333 ± 0.0024 |
| Mixhop | 0.6987 ± 0.0029 | 0.6895 ± 0.0055 | 0.6912 ± 0.0069 |
| SGC | 0.6187 ± 0.0046 | 0.6136 ± 0.0043 | 0.6137 ± 0.0065 |

