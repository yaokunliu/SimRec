import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from utils.utils import get_parser, setup_seed
from utils.evalution import train, test, output

pid = os.getpid()
print('pid:%d' % (pid))

if __name__ == '__main__':
    print(sys.argv)
    parser = get_parser()
    args = parser.parse_args()
    if args.gpu:
        device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
        print("use cuda:"+args.gpu if torch.cuda.is_available() else "use cpu, cuda:"+args.gpu+" not available")
    else:
        device = torch.device("cpu")
        print("use cpu")
    
    SEED = args.random_seed
    setup_seed(SEED)

    if args.dataset == 'Books':
        item_count = 367982 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
        head_thre = 12173
        tail_thre = 74672
    elif args.dataset == 'Beauty':
        item_count = 12101 + 1
        batch_size = 256
        seq_len = 20
        test_iter = 500
        head_thre = 750
        tail_thre = 3451
    elif args.dataset == 'ML_1M':
        item_count = 3416 + 1
        batch_size = 256
        seq_len = 50
        test_iter = 500
        head_thre = 236
        tail_thre = 776
    elif args.dataset == 'Retailrocket':
        item_count = 25310 + 1 # 34185 + 1
        batch_size = 256
        seq_len = 20
        test_iter = 500
        head_thre = 1977 # 2236
        tail_thre = 8029 # 9611
    elif args.dataset == 'Yelp':
        item_count = 14142 + 1 
        batch_size = 256 # 256
        seq_len = 20
        test_iter = 500
        head_thre = 1304 
        tail_thre = 5012
    elif args.dataset == 'Sports':
        item_count = 18357 + 1 
        batch_size = 256
        seq_len = 20
        test_iter = 500
        head_thre = 1151 
        tail_thre = 5458
    elif args.dataset == 'Gowalla':
        item_count = 105892 + 1 
        batch_size = 128
        seq_len = 40
        test_iter = 1000
        head_thre = 7135 
        tail_thre = 36627

    path = './data/' + args.dataset + '/'

    if args.sparse_ratio > 0:
        train_file = path + args.dataset + '_train' + '_' + str(args.sparse_ratio) + '.txt'
    else:
        train_file = path + args.dataset + '_train.txt'

    valid_file = path + args.dataset + '_valid.txt'

    if args.noise_ratio > 0:
        test_file = path + args.dataset + '_test' + '_' + str(args.noise_ratio) + '.txt'
    else:
        test_file = path + args.dataset + '_test.txt'    
    dataset = args.dataset

    print("hidden_size:", args.hidden_size)
    print("interest_num:", args.interest_num)

    if args.p == 'train':
        train(args=args, device=device, train_file=train_file, valid_file=valid_file, test_file=test_file, dataset=dataset, model_type=args.model_type,
              item_count=item_count, batch_size=batch_size, lr=args.learning_rate, 
              seq_len=seq_len, hidden_size=args.hidden_size, interest_num=args.interest_num, topN=args.topN, max_iter=args.max_iter, test_iter=test_iter, 
              decay_step=args.lr_dc_step, lr_decay=args.lr_dc, patience=args.patience, head_thre=head_thre, tail_thre=tail_thre)
    elif args.p == 'test':
        test(args=args, device=device, test_file=test_file, dataset=dataset, model_type=args.model_type, item_count=item_count, batch_size=batch_size, 
             lr=args.learning_rate, seq_len=seq_len, hidden_size=args.hidden_size, interest_num=args.interest_num, topN=args.topN, patience=args.patience, 
             head_thre=head_thre, tail_thre=tail_thre)
    elif args.p == 'output':
        output(args=args, device=device, dataset=dataset, model_type=args.model_type, item_count=item_count, batch_size=batch_size, lr=args.learning_rate, 
               seq_len=seq_len, hidden_size=args.hidden_size, interest_num=args.interest_num, topN=args.topN, patience=args.patience, head_thre=head_thre, tail_thre=tail_thre)
    else:
        print('do nothing...')
    
