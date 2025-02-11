import numpy as np
import pandas as pd
import myfunctions as fn
from args import args, dev
import torch
from torch.utils.data import DataLoader
import proposed_model as model


hg = pd.read_csv("./datasets/SZ247_hypergraph_2.csv",header=0,index_col=0)
hg = np.array(hg) # (nodes, e)
hg_tensor = torch.Tensor(hg).to(dev)

adj = pd.read_csv(args.data_path + 'SZ247_adj.csv', index_col=0, header=0)
adj_dense = np.array(adj, dtype=float)
nodes = adj.shape[0]
adj_dense = torch.Tensor(adj_dense)
adj = adj_dense.to_sparse_coo().to(dev)

fn.seed_torch(2024)
occ, prc, temp = fn.get_data(args)  # (t, nodes)
occ_train, occ_valid, occ_test = fn.division(args, occ)
prc_train, prc_valid, prc_test = fn.division(args, prc)
temp_train, temp_valid, temp_test = fn.division(args, temp)
train_dataset = fn.MyDataset(args, occ_train, prc_train, temp_train, dev)
valid_dataset = fn.MyDataset(args, occ_valid, prc_valid, temp_valid, dev)
test_dataset = fn.MyDataset(args, occ_test, prc_test, temp_test, dev)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False)

train_model = model.proposed_Model(args, hg_tensor,adj).to(dev)
model.training(train_model, train_loader, valid_loader, args)
trained_model = model.proposed_Model(args, hg_tensor,adj).to(dev)
trained_model.load_state_dict(torch.load("checkpoint.pt"))
torch.save(trained_model, "./result_"+str(args.predict_time)+"/proposed_"+args.method+".pt")
output = model.test(trained_model, test_loader, args)
output = np.array(output).reshape(1,6)
result_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'],data=output)
result_df.to_csv("./result_"+str(args.predict_time)+"/proposed_"+args.method+".csv")
