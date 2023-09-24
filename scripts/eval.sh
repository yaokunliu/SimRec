#!/bin/bash
cd $(dirname $0) && cd ..


# eval ComiRec
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type item_emb --dataset Beauty
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type item_emb --dataset Books
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type item_emb --dataset Retailrocket
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type item_emb --dataset Yelp

# eval SimRec (ComiRec + SimEmb)
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 3 --dataset Beauty
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 3 --dataset Books
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 3 --dataset Retailrocket
CUDA_VISIBLE_DEVICES=0 python train.py --p test --model_type ComiRec-SA --emb_type attr_emb --adj_mat_step_size 5 --dataset Yelp
