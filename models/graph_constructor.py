import os
import torch
import pickle
import numpy as np
import scipy.sparse as sp

from time import time
from tqdm import tqdm

np.random.seed(123)


class GraphConstructor(object):
    def __init__(self, source, dataset, item_count, adj_mat_type, adj_mat_step_size, device):
        self.source = source
        self.item_count = item_count
        self.mat_type = adj_mat_type
        self.k = adj_mat_step_size
        self.device = device
        self.path = '../Pseudo_Mat/' + dataset + '_' + self.mat_type + str(self.k) + '.npz'

    def convert_sp_to_tensor(self, matrix):
        coo = sp.coo_matrix(matrix)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)._coalesced_(True)

    # def check_zero_row(self, dok_matrix):
    #     zero_row = []
    #     for i in range(self.item_count):
    #         if dok_matrix[i, :].sum() == 0:
    #             zero_row.append(i)
    #     return zero_row

    def create_adj_mat(self, tqdm_disable=True):
        if os.path.exists(self.path):
            adj_mat = sp.load_npz(self.path)
            print('Loading adjacency matrix from %s done' % (self.path))

        else:
            print('Creating adjacency matrix...')
            if not self.source:
                raise ValueError('No source file to create adjacency matrix')

            start = time()
            if self.mat_type in ['i', 'dfi', 'afi', 'ai']:
                adj_mat = sp.dok_matrix((self.item_count, self.item_count), dtype=np.float32)
            else:
                adj_mat = sp.dok_matrix((self.item_count, self.item_count), dtype=np.int32)
            with open(self.source, 'r') as f:
                for line in tqdm(f.readlines(), disable=tqdm_disable):
                    conts = line.strip().split(' ')
                    seq = conts[1:]
                    for i in range(len(seq)):
                        item_i = int(seq[i])
                        for j in range(i - self.k, i + self.k + 1):
                            if j < 0 or j >= len(seq) or i == j:
                                continue
                            item_j = int(seq[j])
                            # -------------------------a--------------------------------
                            if self.mat_type in ['a', 'ai']:    # 对称
                                adj_mat[item_i, item_j] += 1
                            # -------------------------af-------------------------------
                            elif self.mat_type in ['af', 'afi']:  # 不对称
                                if i > j:
                                    adj_mat[item_i, item_j] += 1
                            # -------------------------ab-------------------------------
                            elif self.mat_type == 'ab':  # 不对称
                                if i > j:
                                    adj_mat[item_i, item_j] += 2
                                else:
                                    adj_mat[item_i, item_j] += 1
                            # -------------------------d--------------------------------
                            elif self.mat_type in ['d','i']:  # 对称
                                adj_mat[item_i, item_j] += self.k - abs(i - j) + 1
                            # -------------------------df-------------------------------
                            elif self.mat_type in ['d','dfi']:  # 不对称
                                if i > j:
                                    adj_mat[item_i, item_j] += self.k - abs(i - j) + 1
                            # -------------------------db-------------------------------
                            elif self.mat_type == 'db':  # 不对称
                                if i > j:
                                    adj_mat[item_i, item_j] += (self.k - abs(i - j) + 1) * 2
                                else:
                                    adj_mat[item_i, item_j] += self.k - abs(i - j) + 1
                            else:
                                raise ValueError('Invalid adjacency matrix type')

            if self.mat_type in ['i','dfi', 'afi', 'ai']:
                degrees = np.array(adj_mat.sum(1))
                d_hat = sp.diags(np.power(degrees, -1).flatten())
                adj_mat = d_hat.dot(adj_mat)
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                adj_mat = adj_mat.tocsr()
            else:
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                degrees = np.array(adj_mat.sum(1))
                d_hat = sp.diags(np.power(degrees, -1).flatten())
                adj_mat = d_hat.dot(adj_mat).tocsr()              

            # zero_row = self.check_zero_row(adj_mat)
            # if len(zero_row) != 0:
            #     raise ValueError('There are zero rows in adj_mat')

            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            sp.save_npz(self.path, adj_mat)
            print('Creating original adjacency matrix done, time elapsed: {:.2f}s'.format(time() - start))

        print('Adjacency matrix shape:', adj_mat.shape, ', Adjacency matrix nnz:', adj_mat.nnz)
        return adj_mat, adj_mat.shape[1]


if __name__ == '__main__':
    dataset = 'Beauty'  # Beauty | Books | ML_1M | Tmall | Retailrocket | Yelp | Sports | Gowalla

    if dataset == 'Books':
        item_count = 367982 + 1
    elif dataset == 'Beauty':
        item_count = 12101 + 1
    elif dataset == 'ML_1M':
        item_count = 3416 + 1
    elif dataset == 'Retailrocket':
        item_count = 25310 + 1
    elif dataset == 'Yelp':
        tem_count = 14142 + 1
    elif dataset == 'Sports':
        item_count = 18357 + 1
    elif dataset == 'Gowalla':
        item_count = 105892 + 1

    source = './data/' + dataset + '/' + dataset + '_train.txt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adj_mat, a = GraphConstructor(source, dataset, item_count, 'i', 3, device).create_adj_mat(tqdm_disable=False)
    print(adj_mat[1])
