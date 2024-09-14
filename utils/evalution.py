import os
import sys
import time
import math
import faiss
import torch
import numpy as np
import pandas as pd

from data.data_iterator import get_DataLoader
from models.graph_constructor import GraphConstructor
from utils.utils import get_exp_name, get_model, load_model, to_tensor


def evaluate(model, test_data, hidden_size, emb_type, attr_weight, device, k=50, head_thre=0, tail_thre=0):

    topN = k # 评价时选取topN
    item_embs = model.output_items(attr_weight).cpu().detach().numpy()

    res = faiss.StandardGpuResources() 
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0 # 指定GPU设备
    
    try:
        gpu_index = faiss.GpuIndexFlatIP(res, hidden_size, flat_config) # 建立GPU index用于Inner Product近邻搜索
        gpu_index.add(item_embs) # 给index添加向量数据
    except Exception as e:
        print("error:", e)
        return

    total = 0
    total_recall, total_ndcg, total_hitrate = 0.0, 0.0, 0.0

    if head_thre != 0:
        total_head, total_medium, total_tail = 0, 0, 0
        total_recall_head, total_hitrate_head = 0.0, 0.0
        total_recall_tail, total_hitrate_tail = 0.0, 0.0
        total_recall_medium, total_hitrate_medium = 0.0, 0.0

    for _, (users, targets, items) in enumerate(test_data): # 一个batch的数据        
        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        user_embs, _ = model(to_tensor(items, device), None, None, attr_weight, device, train=False)
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2: # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list) # 不重复label物品
                for no, iid in enumerate(I[i]): # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
                    if iid in true_item_set: # 如果该推荐物品是label物品
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                
        else: # 多兴趣模型评估
            ni = user_embs.shape[1] # num_interest
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]]) # shape=(batch_size*num_interest, embedding_dim)
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0

                if head_thre != 0:
                    recall_head, recall_medium, recall_tail = 0, 0, 0
                    head_num, medium_num, tail_num = 0, 0, 0

                item_list_set = set()
                item_cor_list = []
                
                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                item_list.sort(key=lambda x:x[1], reverse=True) # 降序排序，内积越大，向量越近
                for j in range(len(item_list)): # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        item_cor_list.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
            
                true_item_set = set(iid_list)

                if head_thre != 0:
                    for tid in true_item_set: 
                        if tid <= head_thre: head_num += 1 
                        elif tid >= tail_thre: tail_num += 1
                        else: medium_num += 1

                for no, iid in enumerate(item_cor_list): # 对于推荐的每一个物品
                    if iid in true_item_set: # 如果该物品是label物品
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                        if head_thre != 0:
                            if iid <= head_thre: recall_head += 1
                            elif iid >= tail_thre: recall_tail += 1
                            else: recall_medium += 1

                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list) # len(iid_list)表示label数量
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                
                if head_thre != 0:
                    if head_num != 0:
                        total_head += 1
                        total_recall_head += recall_head * 1.0 / head_num
                        if recall_head > 0: total_hitrate_head += 1
                    if medium_num != 0:
                        total_medium += 1
                        total_recall_medium += recall_medium * 1.0 / medium_num
                        if recall_medium > 0: total_hitrate_medium += 1
                    if tail_num != 0:
                        total_tail += 1
                        total_recall_tail += recall_tail * 1.0 / tail_num
                        if recall_tail > 0: total_hitrate_tail += 1
        
        total += len(targets) # total增加每个批次的用户数量
    
    recall = total_recall / total # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total # NDCG
    hitrate = total_hitrate * 1.0 / total # 命中率
    
    if head_thre != 0:
        recall_head, recall_medium, recall_tail = total_recall_head / total_head, total_recall_medium / total_medium, total_recall_tail / total_tail
        hitrate_head, hitrate_medium, hitrate_tail = total_hitrate_head * 1.0 / total_head, total_hitrate_medium * 1.0 / total_medium, total_hitrate_tail * 1.0 / total_tail # 命中率
        appendix_dict = {'recall_head': recall_head, 'recall_medium': recall_medium, 'recall_tail': recall_tail, 'hitrate_head': hitrate_head, 'hitrate_medium': hitrate_medium, 'hitrate_tail': hitrate_tail}
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}, appendix_dict
        
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}, None
    
    

def iterate(device, train_data, valid_data, dataset, model, optimizer, scheduler, iter, item_count, batch_size, test_iter, hidden_size, 
            topN, exp_name, best_model_path, patience, trials, best_metric, max_iter, head_thre, tail_thre, emb_type, attr_weight):
    
    start_time = time.time()

    total_loss = 0.0

    metric_path = 'metrics_log/' + exp_name +'/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    
    for i, (users, targets, items) in enumerate(train_data):
        train_time = time.time()
        model.train()
        iter += 1
        optimizer.zero_grad()        
        
        negs = list(set(range(item_count)) ^ set(targets)) 
        negs = np.random.choice(negs, 10 * batch_size, replace=False) # comi原代码负样本个数就是10 * batch_size         

        _, loss = model(to_tensor(items, device), to_tensor(targets, device), to_tensor(negs, device), attr_weight, device) 
        
        if loss.item() > 1e5 or math.isnan(loss.item()): # 损失值过大或为nan，跳过这次迭代
            print("loss is abnormal!")
            continue

        loss.backward()
        optimizer.step()

        # model.print_parameters()
         
        total_loss += loss

        if iter % test_iter == 0: 
            print('-' * 99)
            print("train time one batch: %.4fs" % (time.time()-train_time))   
            # if iter % 1000 == 0 and iter != 0:  
            #     scheduler.step()
            print('lr: %.4f'%(optimizer.param_groups[0]["lr"]))                      

            model.eval()

            metrics, _ = evaluate(model, valid_data, hidden_size, emb_type, attr_weight, device, topN)
            with open (metric_path + 'valid_metrics.txt', 'a') as f:
                f.write('%d\t%.4f\t%.4f\t%.4f\n' % (iter, metrics['recall'], metrics['ndcg'], metrics['hitrate']))
            
            log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / test_iter) 
            if metrics != {}:
                log_str += ', ' + ', '.join(['valid ' + key + ': %.4f' % value for key, value in metrics.items()])
            print(exp_name)
            print(log_str)         
            
            with open (metric_path+'loss.txt',"a") as f:
                f.write('%d\t%.4f\n' % (iter, total_loss / test_iter))

            # 保存recall最佳的模型
            if 'recall' in metrics:
                recall = metrics['recall']
                # recall = metrics['recall'] + metrics['hitrate'] + metrics['ndcg']
                if recall > best_metric:
                    best_metric = recall
                    checkpoint = {
                                'iter': iter,
                                'net': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_metric': best_metric,
                                }
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path)
                    torch.save(checkpoint, best_model_path + 'ckpt.pth')
                    trials = 0
                else:
                    trials += 1
                    if trials > patience: # early stopping
                        print("early stopping!")
                        return iter, best_metric, trials, model, optimizer

            # 每次test之后loss_sum置零
            total_loss = 0.0
            test_time = time.time()
            print("time interval: %.4f min" % ((test_time-start_time)/60.0))
            sys.stdout.flush()
        
        if iter >= max_iter * 1000: # 超过最大迭代次数，退出训练
            return iter, best_metric, trials, model, optimizer    


def train(args, device, train_file, valid_file, test_file, dataset, model_type, item_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, max_iter, test_iter, decay_step, lr_decay, patience, head_thre, tail_thre):  

    exp_name = get_exp_name(dataset, model_type, args.emb_type, args.attr_weight, batch_size, lr, hidden_size, seq_len, interest_num, topN, patience) 
    best_model_path = "./best_model/" + exp_name + '/' 
    
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1)
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0)

    adj_mat = GraphConstructor(train_file, dataset, item_count, args.adj_mat_type, args.adj_mat_step_size, device)
    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, args.temperature, args.emb_type, adj_mat)
    # return
    model = model.to(device)
    
    params_num = sum(p.numel() for p in model.parameters())
    print("Total params: ", params_num)

    optimizer = torch.optim.Adam(model.parameters(), lr) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=lr_decay) 

    iter_ckpt  = 0
    best_metric_ckpt = 0.0

    # 断点续训
    if os.path.exists(best_model_path):  
        path_checkpoint = best_model_path + 'ckpt.pth' 
        checkpoint = torch.load(path_checkpoint)  
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])  
        iter_ckpt = checkpoint['iter']
        best_metric_ckpt = checkpoint['best_metric']
        print("Model restored from:", path_checkpoint)
        print("iter: %d, best_recall: %.4f" %(iter_ckpt, best_metric_ckpt))
    else:
        print('No Model')

    print('training begin')
    sys.stdout.flush()
       
    try:
        trials = 0
        total_loss = 0.0
        iter = iter_ckpt
        best_metric = best_metric_ckpt # 最佳指标值，在这里是最佳recall值

        iter, best_metric, trials, model, optimizer = iterate(device, train_data, valid_data, dataset, model, optimizer, scheduler, iter, item_count, 
                                                          batch_size, test_iter, hidden_size, topN, exp_name, best_model_path, 
                                                          patience, trials, best_metric, max_iter, head_thre, tail_thre, args.emb_type, args.attr_weight)           

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')
    
    load_model(model, best_model_path)
    model.eval()

    # 训练结束后用valid_data测试一次
    print('-' * 99)
    metrics, _ = evaluate(model, valid_data, hidden_size, args.emb_type, args.attr_weight, device, k=20)
    print(', '.join(['valid_20 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    metrics, _ = evaluate(model, valid_data, hidden_size, args.emb_type, args.attr_weight, device, k=50)
    print(', '.join(['valid_50 ' + key + ': %.4f' % value for key, value in metrics.items()]))

    # 训练结束后用test_data测试一次
    print('-' * 99)
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=5)
    print(', '.join(['test_5 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=10)
    print(', '.join(['test_10 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=20)
    print(', '.join(['test_20 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=50)
    print(', '.join(['test_50 ' + key + ': %.4f' % value for key, value in metrics.items()]))


def test(args, device, test_file, dataset, model_type, item_count, batch_size, lr, seq_len, hidden_size, interest_num, topN, patience, head_thre, tail_thre):
    
    exp_name = get_exp_name(dataset, model_type, args.emb_type, args.attr_weight, batch_size, lr, hidden_size, seq_len, interest_num, topN, patience, save=False) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径A

    adj_mat = GraphConstructor(None, dataset, item_count, args.adj_mat_type, args.adj_mat_step_size, device)
    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, args.temperature, args.emb_type, adj_mat)
    load_model(model, best_model_path)
    model = model.to(device)

    model.eval()
    test_time = time.time()
        
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)

    # metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=5)
    # print(', '.join(['test_5 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    # metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=10)
    # print(', '.join(['test_10 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    # metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=20)
    # print(', '.join(['test_20 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    metrics, _ = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, k=50)
    print(', '.join(['test_50 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    print('test time: %.4f' % (time.time() - test_time))

    # metrics, appendix = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, 20, head_thre, tail_thre)
    # print(', '.join(['test_20 ' + key + ': %.4f' % value for key, value in metrics.items()]))
    # print(', '.join([key + ': %.4f' % value for key, value in appendix.items()]))
    # metrics, appendix = evaluate(model, test_data, hidden_size, args.emb_type, args.attr_weight, device, 50, head_thre, tail_thre)
    # print(', '.join(['test_50 ' + key + ': %.4f' % value for key, value in metrics.items()]))    
    # print(', '.join([key + ': %.4f' % value for key, value in appendix.items()]))



def output(args, device, dataset, model_type, item_count, batch_size, lr, seq_len, hidden_size, interest_num, topN, patience, head_thre, tail_thre):
    
    exp_name = get_exp_name(dataset, model_type, args.emb_type, args.attr_weight, batch_size, lr, hidden_size, seq_len, interest_num, topN, patience, save=False) # 实验名称
    exp_name = exp_name[:-2] + str(args.adj_mat_type) + str(args.adj_mat_step_size)
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径
    
    adj_mat = GraphConstructor(None, dataset, item_count, args.adj_mat_type, args.adj_mat_step_size, device)
    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, args.temperature, args.emb_type, adj_mat)
    load_model(model, best_model_path)
    model = model.to(device)

    if args.emb_type != 'item_emb':
        output_path = 'output/' + exp_name + '_attr/'  
    else:
        output_path = 'output/' + exp_name + '_item/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    npfile_path = output_path  + exp_name + '_emb.npy'
    tsvfile_path = output_path  + exp_name + '_emb.tsv'

    model.eval()

    item_embs = model.output_items(args.attr_weight).cpu().detach().numpy() # 获取物品嵌入
    np.save(npfile_path, item_embs) # 保存物品嵌入

    # npfile = np.load(npfile_path)
    # np_to_csv = pd.DataFrame(data = npfile)
    # np_to_csv.to_csv(tsvfile_path, sep='\t', header=False, index=False, mode="w") # 输出为tsv文件

