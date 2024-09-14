#!/bin/bash
cd $(dirname $0) && cd ..


# train ComiRec
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type item_emb --dataset Beauty
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type item_emb --dataset Books
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type item_emb --dataset Retailrocket
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type item_emb --dataset Yelp

# train SimRec (ComiRec + SimEmb)
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 3 --dataset Beauty
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 3 --dataset Books
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 3 --dataset Retailrocket
CUDA_VISIBLE_DEVICES=0 python train.py --p train --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 5 --dataset Yelp
