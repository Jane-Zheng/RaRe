#!/bin/bash
# cd ./project/RAG-test/RL/PPO
# nohup bash run_online_ppo.sh
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

python online_ppo_topiocqa.py \
  --seed 42 \
  --train_file ./project/RAG-test/topicoqa/papers_full_train_re.jsonl \
  --val_file ./project/RAG-test/topicoqa/devdata/select_papers.jsonl \
  --cache_dir ./project/RAG-test/RL/PPO/cache \
  --base_model_name ./project/RAG-test/Flan-T5-large \
  --policy_model_name ./project/RAG-test/t5-checkpoint/sft_flant5_large_redata_new_topiocqa_gpt_0408_2/checkpoint-8520 \
  --generator_model_path ./project/RAG-test/Meta-Llama-3-8B-Instruct \
  --output_dir ./project/RAG-test/RL/PPO/checkpoint/online_ppo_topiocqa_m9r1_0416_1 \
  --logging_dir ./project/RAG-test/RL/PPO/checkpoint/online_ppo_topiocqa_m9r1_0416_1/logs \
  --retriever_collection_tsv ./project/Llama2/datasets/topiocqa/selected_wiki_segments.tsv \
  --retriever_model_path ./project/RAG-test/msmarco-roberta-base-ance-firstp \
  --retriever_index_path ./project/RAG-test/first_part_test/select_train_topiocqa_faiss_cos.index \
  --retriever_topk 5 \
  --generator_max_docs 1 \
  --retriever_embed_batch_size 32 \
  --max_ctx_length 512 \
  --max_suffix_length 64 \
  --max_new_tokens 64 \
  --ppo_batch_size 16 \
  --mini_batch_size 8 \
  --ppo_epochs 2 \
  --total_steps 3000 \
  --lr 5e-6 \
  --save_freq 200 \
  --log_freq 100 \
  --mrr_weight 0.9 \
  --rouge_weight 0.1 \
  --normalize_mrr \
  --num_samples_per_query 2 \
  --kl_coef 0.05 \
  --cliprange 0.2 \
  --cliprange_value 0.2 \
  --vf_coef 0.2 \
  --policy_device cuda:1 \
  --reward_device cuda:0 \
  --temperature 0.5 \
  --top_p 0.9 \
  --bf16
