import torch
import random
from torch.utils.data import DataLoader

class DataIterator(torch.utils.data.IterableDataset):

    def __init__(self, 
                 source,
                 batch_size,
                 seq_len,
                 train_flag,         
                ):
        self.read(source) 
        self.users = list(self.users) 
        self.batch_size = batch_size # 用于训练
        self.eval_batch_size = batch_size # 用于验证、测试
        self.train_flag = train_flag # train_flag=1表示训练
        self.seq_len = seq_len # 历史物品序列的最大长度
        self.index = 0 # 验证和测试时选择用户的位置的标记
        
        if self.train_flag==1:
            print("user num for train:", len(self.users))
        else:
            print("user num for eval:", len(self.users))
            
    def __iter__(self):
        return self

    def read(self, source):
        self.graph = {} # key:user_id，value:一个list，放着该user_id所有(item_id,time_stamp)元组，排序后value只保留item_id
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(' ')
                user_id = int(conts[0])
                self.users.add(user_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                for item_id in conts[1:]:
                    item_id=int(item_id)
                    self.items.add(item_id)
                    self.graph[user_id].append(item_id)       
        self.users = list(self.users) # 用户列表
        self.items = list(self.items) # 物品列表
    
    def __next__(self):
        if self.train_flag == 1: # 训练
            user_id_list = random.sample(self.users, self.batch_size) # 随机抽取batch_size个user
        else: # 验证、测试，按顺序选取eval_batch_size个user，直到遍历完所有user
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_item_list = []

        for user_id in user_id_list:
            item_list = self.graph[user_id] # 排序后的user的item序列
            if self.train_flag == 1: # 训练，选取训练时的label
                k = random.choice(range(4, len(item_list))) # 从[4,len(item_list))中随机选择一个index
                item_id_list.append(item_list[k]) # 该index对应的item加入item_id_list
            else: # 验证、测试，选取该user后20%的item用于验证、测试
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])

            # k前的item序列为历史item序列
            if k >= self.seq_len: # 选取seq_len个物品
                hist_items = item_list[k-self.seq_len: k]
                hist_item_list.append(hist_items)
                # hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_items = item_list[:k]
                hist_item_list.append(hist_items + [0] * (self.seq_len - k))
                # hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
                
        # 返回用户列表（batch_size）、物品列表（label）（batch_size）、
        # 历史物品列表（batch_size，seq_len）、历史物品的mask列表（batch_size，seq_len）(暂不输出)
        # 历史增强列表（batch_size，2，seq_len）, 训练或不利用自监督时，输出为空列表
        return user_id_list, item_id_list, hist_item_list

def get_DataLoader(source, batch_size, seq_len, train_flag):
    dataIterator = DataIterator(source, batch_size, seq_len, train_flag)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)

