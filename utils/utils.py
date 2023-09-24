import os
import torch
import random
import shutil
import argparse
import numpy as np
from models.ComiRec import ComiRec_SA

def get_parser():    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--p', type=str, default='train', help='train | test | output')  # TODO
    parser.add_argument('--dataset', type=str, default='Beauty', help='Beauty | Books | ML_1M | Tmall | Retailrocket | Yelp | Sports | Gowalla')  # TODO
    parser.add_argument('--random_seed', type=int, default=2021)  # TODO
    parser.add_argument('--hidden_size', type=int, default=64) 
    parser.add_argument('--interest_num', type=int, default=4) 
    parser.add_argument('--topN', type=int, default=50) 

    parser.add_argument('--gpu', type=str, default='0', help='None -> cpu') 
    parser.add_argument('--model_type', type=str, default='ComiRec-SA', help='ComiRec-SA') 
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate (default=0.001)') 
    parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')  
    parser.add_argument('--lr_dc_step', type=int, default=1000, help='(k), the number of steps after which the learning rate decay') 
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)') 
    parser.add_argument('--patience', type=int, default=50)    
    
    parser.add_argument('--temperature', default=1.0, type=float, help='infoNCE temperature (default=1.0)') 
    parser.add_argument('--emb_type', default='attr_emb', type=str, help='sum_emb | attr_emb | item_emb | concate_emb') # TODO
    parser.add_argument('--attr_weight', default=1, type=float) # TODO
    parser.add_argument('--sparse_ratio', default=0, type=float) # TODO
    parser.add_argument('--noise_ratio', default=0, type=float) # TODO 

    parser.add_argument('--adj_mat_type', default='d', type=str, help='type of the adjacency matrix') # TODO
    parser.add_argument('--adj_mat_step_size', default=3, type=int, help='the window length of the adjacency matrix') # TODO

    return parser

# 生成实验名称
def get_exp_name(dataset, model_type, emb_type, attr_weight, batch_size, lr, hidden_size, seq_len, interest_num, topN, patience=50, save=True):
    # extr_name = input('Please input the experiment name: ')
    extr_name = 'd3'  # TODO
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(hidden_size), 'len'+str(seq_len), 
                          'in'+str(interest_num), 'top'+str(topN), 'pat'+str(patience), emb_type+str(attr_weight)])
    exp_name = para_name + '_' + extr_name

    # while os.path.exists('best_model/' + exp_name) and save:
    #     flag = input('The exp name already exists. Do you want to cover? (y/n)')
    #     if flag == 'y' or flag == 'Y':
    #         shutil.rmtree('best_model/' + exp_name) # 删除原模型
    #         break
    #     else:
    #         extr_name = input('Please input the experiment name: ')
    #         exp_name = para_name + '_' + extr_name

    return exp_name

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 获取模型
def get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, temperature, emb_type, adj_mat = None):
    if model_type == 'ComiRec-SA':
        model = ComiRec_SA(item_count, hidden_size, batch_size, interest_num, seq_len, adj_mat, temperature, emb_type, add_pos=True)
    else:
        print("Invalid model_type : %s", model_type)
        return
    
    return model

def save_model(model, Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')

def load_model(model, path):
    path_checkpoint = path + 'ckpt.pth' # restore_model_path
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint['net'])
    print('model loaded from %s' % path)

def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()



