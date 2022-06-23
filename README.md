# dgraph_cogdl
cogdl version of Dgraph

This repo provides a collection of cogdl baselines for DGraphFin dataset. Please download the dataset from the DGraph web and place it under the folder 'dataset/'

**Dgraph dataset:** https://dgraph.oss-cn-shanghai.aliyuncs.com/DGraphFin.zip

**Cogdl introduction:** https://cogdl.readthedocs.io/en/latest/index.html

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

- **Other Parameters**
```bash
python gnn.py --model gcn --epochs 200 --runs 10 --device 0 --hidden_size 128 --lr 0.01 --dropout 0.5 --early_stop False
```


## Results:
Performance on **DGraphFin**(10 runs):

| Methods   | Valid AUC  | Test AUC  |
|  :----  |  ---- | ---- |
| MLP | 0.6987 ± 0.0029 | 0.7059 ± 0.0030 |
| GCN | 0.7093 ± 0.0198 | 0.7115 ± 0.0205 |
| GraphSAGE| 0.7521 ± 0.0021 | **0.7601 ± 0.0013** |
| Grand  | 0.6935 ± 0.0025 | 0.6955 ± 0.0020 |
| GAT  | 0.7233 ± 0.0012 | 0.7333 ± 0.0024 |
| Mixhop | 0.6895 ± 0.0055 | 0.6912 ± 0.0069 |
| SGC | 0.6436 ± 0.0443 | 0.6437 ± 0.0465 |
