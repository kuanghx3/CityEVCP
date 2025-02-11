import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from tqdm import tqdm
from args import dev
import myfunctions as fn
import pandas as pd
from pytorchtools import EarlyStopping

class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout=0, alpha=0.2):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=dev))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1, device=dev)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        # sparse matrix
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat

class MHGHAtten(nn.Module):
    def __init__(self, hg, head_n, input_dim=12, out_dim=12, dropout=0, alpha=0.2):
        super(MHGHAtten, self).__init__()

        # step 1 init
        self.hg = hg
        idx = torch.nonzero(self.hg).T
        data = self.hg[idx[0], idx[1]]
        coo_hg = torch.sparse_coo_tensor(idx, data, self.hg.shape)
        self.hg_indices = coo_hg.coalesce().indices()
        self.values = coo_hg.coalesce().values()
        self.N = self.hg.shape[0]
        self.P = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
        nn.init.xavier_normal_(self.P, gain=1.414)

        #step 2 init
        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=dev))
            self.heads_dict[n, 2] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 2], gain=1.414)
        self.linear = nn.Linear(head_n, 1)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        hg_dense = copy.deepcopy(hg)
        hg_dense[torch.where(hg_dense == 0)] = -1000000000
        hg_dense[torch.where(hg_dense == 1)] = 0
        self.mask = hg_dense

        self.linear2 = nn.Linear(12,1)
        self.gcn = nn.Linear(in_features=12, out_features=12)

    def forward(self, x):
        b, s, n = x.shape
        x = x.permute(0,2,1)  # (b,n,s)
        e = self.leakyrelu(torch.matmul(self.hg.T, torch.matmul(x, self.P)))  # (b,e,s)
        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            x_h = torch.matmul(x, self.heads_dict[n, 0])
            e_h = torch.matmul(e, self.heads_dict[n, 2])
            edge_h = torch.cat((x_h[0,self.hg_indices[0, :], :], e_h[0,self.hg_indices[1, :], :]), dim=1).T
            atts = torch.matmul(self.heads_dict[n, 1], edge_h).squeeze()  # (b, e)
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts).squeeze()
        atts_mat = torch.sparse_coo_tensor(self.hg_indices, mt_atts).to_dense()+self.mask
        atts_mat = self.softmax(atts_mat)

        occ_conv1 = torch.matmul(atts_mat, e)  # (b, n, s)
        occ_conv1 = self.dropout(self.leakyrelu(self.gcn(occ_conv1)))
        y = occ_conv1

        return y

class GRN(nn.Module):
    def __init__(self, input_dim):
        super(GRN, self).__init__()
        self.dense1 = nn.Linear(in_features=input_dim,out_features=input_dim)
        self.elu = nn.ELU()
        self.dense2 = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.dropout = nn.Dropout(0.2)
        self.glu = nn.GLU()
        self.dense3 = nn.Linear(in_features=input_dim, out_features=input_dim)

    def forward(self, x):
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x1 = torch.stack([x,x],dim=-1)
        x1 = self.glu(x1).squeeze(-1)
        x1 = self.dense3(x1)
        ln = nn.LayerNorm(x1.size()[1:], elementwise_affine=True, device=dev)
        y = ln(x1+x)

        return y

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.GRN1 = GRN(input_dim=args.LOOK_BACK)
        self.GRN2 = GRN(input_dim=args.LOOK_BACK)
        self.GRN3 = GRN(input_dim=args.LOOK_BACK)
        self.GRN4 = GRN(input_dim=args.LOOK_BACK)


    def forward(self, x):
        b, s, n = x.shape
        x = x.permute(0,2,1)
        K = self.GRN1(x)
        Q = self.GRN2(x).permute(0, 2, 1)
        V = self.GRN3(x)
        alpha = torch.matmul(Q, K) / K.shape[2]
        alpha = F.gumbel_softmax(alpha, tau=2.5, dim=-2)
        # alpha = F.softmax(alpha, dim=-1)
        x1 = torch.matmul(V, alpha)
        ln = nn.LayerNorm(x1.size()[1:], elementwise_affine=True, device=dev)
        y = ln(x1 + x)
        y1 = self.GRN4(y)
        y = ln(y1 + y)

        return y, alpha

class Encoder2(nn.Module):
    def __init__(self, args):
        super(Encoder2, self).__init__()
        self.GRN1 = GRN(input_dim=args.LOOK_BACK)
        self.GRN2 = GRN(input_dim=args.LOOK_BACK)
        self.GRN3 = GRN(input_dim=args.LOOK_BACK)
        self.GRN4 = GRN(input_dim=args.LOOK_BACK)


    def forward(self, x):
        b, s, n = x.shape
        x = x.permute(0,2,1)
        K = self.GRN1(x)
        Q = self.GRN2(x).permute(0, 2, 1)
        V = self.GRN3(x)
        alpha = torch.matmul(Q, K) / K.shape[2]
        # alpha = F.gumbel_softmax(alpha, tau=1.5, dim=-2)
        alpha = F.softmax(alpha, dim=-1)
        x1 = torch.matmul(V, alpha)
        ln = nn.LayerNorm(x1.size()[1:], elementwise_affine=True, device=dev)
        y = ln(x1 + x)
        y1 = self.GRN4(y)
        y = ln(y1 + y)

        return y, alpha


class Temporal(nn.Module):
    def __init__(self, args):
        super(Temporal, self).__init__()
        self.GRN1 = GRN(input_dim=3)
        self.GRN2 = GRN(input_dim=3)
        self.v_softmax = nn.Softmax(dim=3)
        self.encoder1 = Encoder(args)
        self.encoder2 = Encoder(args)
        self.encoder3 = Encoder(args)
        self.encoder4 = Encoder(args)
        self.encoder5 = Encoder(args)
        self.dense = nn.Linear(in_features=12, out_features=1)

    def forward(self, occ, prc, temp):
        b, s, n = occ.shape
        fea = torch.stack([occ, prc, temp], dim=3)
        fea1 = self.GRN1(fea)
        fea2 = self.GRN2(fea)
        sm_fea2 = self.v_softmax(fea2)
        fea = torch.sum(fea1*sm_fea2, dim=-1)  # (b,s,n)
        y, alpha1 = self.encoder1(fea)
        y1 = y.permute(0,2,1)
        y2, alpha2 = self.encoder2(y1)
        # y2 = y2.permute(0, 2, 1)
        # y2, alpha2 = self.encoder3(y2)
        # y2 = y2.permute(0, 2, 1)
        # y2, alpha2 = self.encoder4(y2)
        # y2 = y2.permute(0, 2, 1)
        # y2, alpha2 = self.encoder5(y2)
        ln = nn.LayerNorm(y.size()[1:], elementwise_affine=True, device=dev)
        y = ln(y+y2)
        y = self.dense(y)
        return y


class proposed_Model(nn.Module):
    def __init__(self, args, hg, adj):
        super(proposed_Model, self).__init__()
        self.seq = args.LOOK_BACK
        self.MHGHAtten = MHGHAtten(hg, head_n=1)
        self.gat_lyr = MultiHeadsGATLayer(adj, self.seq, self.seq, head_n=1)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)
        self.linear2 = nn.Linear(12, 1)
        self.temporal = Temporal(args)
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, prc, temp):
        b, s, n = occ.shape
        occ_hg = self.MHGHAtten(occ).permute(0,2,1)
        occ2 = occ.permute(0,2,1)
        atts_mat = self.gat_lyr(occ2)  # 注意力矩阵 dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, occ2)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1))).permute(0,2,1)
        fea = (occ_conv1 + occ_hg + occ)/3
        y = self.temporal(fea, prc, temp)
        y = y.squeeze(-1)

        return y


def training(model, train_loader, valid_loader, args):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.MSELoss()
    loss_output = []
    valid_losses = []

    for epoch in tqdm(range(args.max_epochs)):
        train_losses = []
        vsm_all = []
        alpha1_all = []
        alpha2_all = []
        for t, data in enumerate(train_loader):
            occ, prc, temp, label = data
            optimizer.zero_grad()
            predict = model(occ, prc, temp)
            # vsm_all.append(vsm.mean(dim=0).cpu().detach().numpy())
            # alpha1_all.append(alpha1.mean(dim=0).cpu().detach().numpy())
            # alpha2_all.append(alpha2.mean(dim=0).cpu().detach().numpy())
            loss = loss_function(predict, label)
            train_losses.append((loss.item()))
            loss.backward()
            optimizer.step()
        train_loss = np.average(train_losses)
        loss_output.append(train_loss)
        epoch_len = len(str(args.max_epochs))
        print_msg = (f'[{epoch + 1:>{epoch_len}}/{args.max_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.8f} ')

        model.eval()  # prep model for evaluation
        for t, data in enumerate(valid_loader):
            occ, prc, temp, vlabel = data
            with torch.no_grad():
                valid_predict = model(occ, prc, temp)
            valid_loss = loss_function(valid_predict, vlabel)
            valid_losses.append(valid_loss.item())

        vloss = np.average(valid_losses)
        valid_losses = []
        early_stopping(vloss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # fn.precess_output(vsm_all, alpha1_all, alpha2_all, args)
            # pd.DataFrame(data=loss_output).to_csv("./result_" + str(args.predict_time) + "/" + "loss_"+args.method+".csv")
            break

        if (epoch + 1) % 100 == 0:
            print(print_msg)
            print("valid_loss=")
            print(vloss)

        # if epoch + 1 == args.max_epochs:
            # fn.precess_output(vsm_all, alpha1_all, alpha2_all, args)
            # pd.DataFrame(data=loss_output).to_csv("./result_" + str(args.predict_time) + "/" + "loss_"+args.method+".csv")

def test(model, test_loader, args):
    approach = "proposed"
    predict_list = []
    label_list = []
    model.eval()

    for t, data in enumerate(test_loader):
        occ, prc, temp, label = data
        with torch.no_grad():
            predict = model(occ,prc,temp)
        predict_list.append(predict.cpu().detach().numpy())
        label_list.append(label.cpu().detach().numpy())
    predict_np = np.concatenate(predict_list,axis=0)
    label_np = np.concatenate(label_list,axis=0)
    np.save("./result_"+str(args.predict_time)+"/proposed_"+args.method+".npy",predict_np)
    metrics = fn.get_metrics(predict_np, label_np)
    print(metrics)
    return metrics

