import os
import torch
from torch import nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
import copy




def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_data(args):
    occ = pd.read_csv(args.data_path+'Shenzhen_Occupancy.csv', index_col=None, header=0)
    inf = pd.read_csv(args.data_path+'Shenzhen_Pile_Information.csv', index_col=None, header=0)
    prc = pd.read_csv(args.data_path+'Shenzhen_Price.csv', index_col=None, header=0)
    temp = pd.read_csv(args.data_path+"temperature_data.csv", index_col=0, header=0)["TEMPERATURE"]
    capability = np.array(inf['count'], dtype=float).reshape(1, -1)
    occ = np.array(occ.iloc[:, 1:], dtype=float) / capability
    prc = np.array(prc.iloc[:, 1:], dtype=float) / 1.5
    temp = np.array(temp)
    temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
    temp = np.expand_dims(temp, axis=1)
    temp = np.repeat(temp, occ.shape[1],axis=1)

    return occ, prc, temp

def division(args, data):

    data_length = len(data)
    train_division_index = int(data_length * args.training_rate)
    train_data = data[:train_division_index, :]
    valid_division_index = int(data_length * (args.training_rate+args.valid_rate))
    valid_data = data[train_division_index:valid_division_index, :]
    test_data = data[valid_division_index:, :]

    return train_data, valid_data, test_data


def create_interval_dataset(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - 2 * lookback):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])

    return np.array(x), np.array(y)

class MyDataset(Dataset):
    def __init__(self, args, occ, prc, temp, dev):
        _, nodes = occ.shape
        occ, label = create_interval_dataset(occ, args.LOOK_BACK, args.predict_time)
        prc, prc_l = create_interval_dataset(prc, args.LOOK_BACK, args.predict_time)
        temp, temp_l = create_interval_dataset(temp, args.LOOK_BACK, args.predict_time)
        self.occ = torch.Tensor(occ)
        self.label = torch.Tensor(label)
        self.occ = torch.Tensor(occ)
        self.prc = torch.Tensor(prc)
        self.temp = torch.Tensor(temp)
        self.device = dev

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):  # occ: batch, seq, node
        return self.occ[idx, :, :].to(self.device), self.prc[idx, :, :].to(self.device), self.temp[idx, :, :].to(self.device), self.label[idx, :].to(self.device)

def get_metrics(test_pre, test_real):
    eps = 0.01
    MAPE_test_real = copy.deepcopy(test_real)
    MAPE_test_pre = copy.deepcopy(test_pre)
    MAPE_test_real[np.where(MAPE_test_real == 0)] = MAPE_test_real[np.where(MAPE_test_real == 0)] + eps
    MAPE_test_pre[np.where(MAPE_test_real == 0)] = MAPE_test_pre[np.where(MAPE_test_real == 0)] + eps
    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))

    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))

    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list

def precess_output(vsm, alpha1, alpha2, args):
    vsm = np.array(vsm)
    vsm = np.average(np.average(np.average(vsm, axis=0), axis=0),axis=0)
    alpha1 = np.array(alpha1)
    alpha2 = np.array(alpha2)
    # alpha = alpha1+alpha2
    alpha1 = np.average(np.average(alpha1, axis=0),axis=1)
    alpha2 = np.average(np.average(alpha2, axis=0),axis=1)
    # pd.DataFrame(data=vsm).to_csv("./result_n6_" + str(args.predict_time) + "/" + "vsm_"+args.method+".csv")
    # pd.DataFrame(data=alpha1).to_csv("./result_n_" + str(args.predict_time) + "/" + "alpha1_"+args.method+".csv")
    # pd.DataFrame(data=alpha2).to_csv("./result_n_" + str(args.predict_time) + "/" + "alpha2_" + args.method + ".csv")

def drop_area(occ, prc, temp, adj, hg):
    n_node = occ.shape[1]
    drop_idx = np.random.randint(n_node, size=int(n_node*0.2))
    occ = np.delete(occ, drop_idx, axis=1)
    prc = np.delete(prc, drop_idx, axis=1)
    temp = np.delete(temp, drop_idx, axis=1)
    adj = np.delete(adj, drop_idx, axis=1)
    adj = np.delete(adj, drop_idx, axis=0)
    hg = np.delete(hg, drop_idx, axis=0)

    return occ, prc, temp, adj, hg