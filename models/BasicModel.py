import os
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.sparse as tp
import torch.nn.functional as F


class BasicModel(nn.Module):

    def __init__(self, item_num, hidden_size, batch_size, seq_len, adj_mat, temperature, emb_type):
        super(BasicModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.item_num = item_num
        self.seq_len = seq_len

        self.device = adj_mat.device
        self.sp_adj_mat, self.att_num = adj_mat.create_adj_mat()
        self.tp_adj_mat = self.convert_sp_to_tp(self.sp_adj_mat, self.device)
        if self.item_num < 1e5:
            self.tp_adj_mat = self.tp_adj_mat.to_dense()
            self.use_sparse_embedding = False
        else:
            self.use_sparse_embedding = True

        self.emb_type = emb_type

        if self.emb_type in ['attr_emb', 'sum_emb']:
            self.attr_embeddings = nn.Embedding(self.att_num, self.hidden_size)
        if self.emb_type in ['item_emb', 'sum_emb']:
            self.item_embeddings = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        if self.emb_type == 'concate_emb':
            self.attr_embeddings = nn.Embedding(self.att_num, int(self.hidden_size/2))
            self.item_embeddings = nn.Embedding(self.item_num, int(self.hidden_size/2), padding_idx=0)

        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def convert_sp_to_tp(self, sp_sparse_mat, device):
        sp_sparse_mat = sp.coo_matrix(sp_sparse_mat)
        coo_data = torch.tensor(sp_sparse_mat.data, dtype=torch.float32)
        coo_indices = torch.tensor(np.vstack((sp_sparse_mat.row, sp_sparse_mat.col)), dtype=torch.int64)
        coo_shape = torch.Size(sp_sparse_mat.shape)
        tp_sparse_mat = torch.sparse_coo_tensor(coo_indices, coo_data, coo_shape).to(device)
        return tp_sparse_mat

    def print_parameters(self):
        print("*"*99)
        print("Parameters:")
        for idx, m in enumerate(self.parameters()):
            print(idx, '->', m)
            print(m.shape)
        print("*"*99)
        print("Modules:")
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)
        print("*"*99)

    def reset_parameters(self, initializer=None):  # TODO,这样可以吗
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)

    @property
    def attr_emb(self):
        if not self.use_sparse_embedding:  # XXX: use_sparse_embedding = False
            attr_emb = torch.matmul(self.tp_adj_mat, self.attr_embeddings.weight)
        else:  # XXX: use_sparse_embedding = True
            attr_emb = tp.mm(self.tp_adj_mat, self.attr_embeddings.weight)
        return attr_emb

    def output_items(self, attr_weight):
        if self.emb_type == 'item_emb':
            return self.item_embeddings.weight

        elif self.emb_type == 'attr_emb':
            return self.attr_emb

        elif self.emb_type == 'sum_emb':
            return self.item_embeddings.weight + attr_weight * self.attr_emb

        elif self.emb_type == 'concate_emb':
            return torch.cat((self.item_embeddings.weight, self.attr_emb), dim=-1)

    def sparse_embedding(self, sparse_mat, indices, device):
        indices_flat = indices.view(-1).cpu()  # 将输入的索引扁平化成一维
        values_flat = sparse_mat[indices_flat]  # 根据索引取出对应的值
        values_flat = self.convert_sp_to_tp(values_flat, device)  # 转换成torch sparse tensor
        return values_flat

    def get_attr_embeddings(self, item_list):
        if not self.use_sparse_embedding:  # XXX: use_sparse_embedding = False
            item_att = F.embedding(item_list, self.tp_adj_mat)  # FIXME: self.tp_adj_mat本身就没有梯度，不需要设置padding_idx=0？
            item_eb = torch.matmul(item_att, self.attr_embeddings.weight)

        else:  # XXX: use_sparse_embedding = True
            item_att = self.sparse_embedding(self.sp_adj_mat, item_list, device=self.device)
            item_eb = tp.mm(item_att, self.attr_embeddings.weight).view(*item_list.shape, -1)

        return item_eb

    def get_item_embeddings(self, item_list, attr_weight):
        if self.emb_type == 'item_emb':
            item_eb = self.item_embeddings(item_list)

        elif self.emb_type == 'attr_emb':
            item_eb = self.get_attr_embeddings(item_list)

        elif self.emb_type == 'sum_emb':
            item_eb = self.get_attr_embeddings(item_list)
            item_eb = self.item_embeddings(item_list) + attr_weight * item_eb

        elif self.emb_type == 'concate_emb':
            item_eb = self.get_attr_embeddings(item_list)
            item_eb = torch.cat((self.item_embeddings(item_list), item_eb), dim=-1)
        return item_eb

    def read_out(self, user_eb, label_eb):
        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        atten = torch.matmul(user_eb,  # shape=(batch_size, interest_num, hidden_size)
                             torch.reshape(label_eb, (-1, self.hidden_size, 1))  # shape=(batch_size, hidden_size, 1)
                             )  # shape=(batch_size, interest_num, 1)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1), dim=-1)  # shape=(batch_size, interest_num)

        if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0], device=user_eb.device) * self.interest_num).long()]
        else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)),  # shape=(batch_size, 1, interest_num)
                                   user_eb  # shape=(batch_size, interest_num, hidden_size)
                                   )  # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
        # readout是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
        return readout  # shape=(batch_size, hidden_size)

    def sampled_softmax(self, user_eb, label_eb, neg_eb, t=1):  # user_eb 即 readout
        # * 是矩阵对位乘法——>(batch_size, embedding_dim), 按dim=1相加是对应两两向量的点积——>(batch_size,1), len(user_eb)==batch_size
        target_embedding = torch.sum(label_eb * user_eb, dim=1).view(len(user_eb), 1)
        product = torch.matmul(user_eb, torch.transpose(neg_eb, 0, 1))  # shape=(batch_size, 10*batch_size), 第(m,n)位置代表的是第m个user_emb和第n个负样本的点积
        # product = torch.cat([target_embedding, product], dim=1)
        loss = torch.exp(target_embedding / t) / (
            torch.sum(torch.exp(product / t), dim=1, keepdim=True) + torch.exp(target_embedding))  # shape=(batch_size, 1)
        loss = torch.mean(-torch.log(loss))
        return loss

    # def NCELoss(self, batch_sample_one, batch_sample_two, device):
    #     sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature # (batch， batch) 第一组增强序列和同组其他序列之间的相似度，(m,n)表示第一组增强序列中第m个序列和第n个序列结果的相似度
    #     sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature # (batch， batch)
    #     sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature # (batch， batch)
    #     d = sim12.shape[-1] # batch
    #     sim11[..., range(d), range(d)] = float('-inf') # 对角线元素全部为float('-inf')，分母不包括和自己的相似度
    #     sim22[..., range(d), range(d)] = float('-inf')
    #     raw_scores1 = torch.cat([sim12, sim11], dim=-1) # (batch, 2*batch) 第一个增强序列和其他2N-1个序列的相似度（注意其中包括正样本），N=b
    #     raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1) # (batch, 2*batch) 第二个增强序列和其他2N-1个序列的相似度（注意其中包括正样本），顺序要和上一个一致，都是先第二组后第一组
    #     logits = torch.cat([raw_scores1, raw_scores2], dim=-2) # (2*batch, 2*batch)
    #     labels = torch.arange(2 * d, dtype=torch.long, device=device) # (2*batch)，label对应logits中横坐标实际正对的纵坐标位置 TODO 改一下device
    #     nce_loss = self.criterion(logits, labels)
    #     return nce_loss


class Multi_Head_Attention_Network(nn.Module):

    def __init__(self, hidden_size, seq_len, num_heads=4, add_pos=True):
        super(Multi_Head_Attention_Network, self).__init__()

        self.hidden_size = hidden_size  # h
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.add_pos = add_pos

        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
            nn.Tanh()
        )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)

    def forward(self, item_eb, mask, device):
        # 历史物品嵌入序列，shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # 矩阵A，shape=(batch_size, num_heads, maxlen)

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(item_att_w,  # shape=(batch_size, num_heads, maxlen)
                                    item_eb  # shape=(batch_size, maxlen, embedding_dim)
                                    )  # shape=(batch_size, num_heads, embedding_dim)

        return interest_emb
