import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BasicModel import BasicModel, Multi_Head_Attention_Network

class ComiRec_SA(BasicModel):
    
    def __init__(self, item_num, hidden_size, batch_size, interest_num, seq_len, adj_mat, temperature, emb_type, add_pos=True):
        super(ComiRec_SA, self).__init__(item_num, hidden_size, batch_size, seq_len, adj_mat, temperature, emb_type)
        self.interest_num = interest_num
        self.num_heads = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        self.multi_head_attention_network = Multi_Head_Attention_Network(self.hidden_size, self.seq_len, self.num_heads, self.add_pos)
        self.reset_parameters()
    
    def get_user_emb(self, item_list, device, attr_weight):
        # 生成mask列表，历史物品列表（batch_size，seq_len），历史物品的mask列表（batch_size，seq_len）
        mask_list = torch.ones_like(item_list, dtype=torch.long)
        mask_list = mask_list.to(device)
        mask_list = torch.where(torch.eq(item_list, 0), item_list, mask_list)
        
        item_eb = self.get_item_embeddings(item_list, attr_weight)
        item_eb = item_eb * torch.reshape(mask_list, (-1, self.seq_len, 1))
        # 用户多兴趣向量
        user_eb = self.multi_head_attention_network(item_eb, mask_list, device) # shape=(batch_size, num_heads, embedding_dim)

        return user_eb
    
    def forward(self, item_list, label_list, neg_list, attr_weight, device, train=True): 
        # 用户多兴趣向量
        user_eb = self.get_user_emb(item_list, device, attr_weight) # shape=(batch_size, num_heads, embedding_dim)

        if not train:
            return user_eb, None 

        label_eb = self.get_item_embeddings(label_list, attr_weight)
        readout = self.read_out(user_eb, label_eb)     
        
        neg_eb = self.get_item_embeddings(neg_list, attr_weight) # shape=(10*batch_size, embedding_dim)
        rec_loss = self.sampled_softmax(readout, label_eb, neg_eb)
            
        return user_eb, rec_loss

        
