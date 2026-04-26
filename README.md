# Online PPO for Conversational Query Rewriting with RAG

This repository implements an online reinforcement learning framework (PPO) for conversational query rewriting, optimized with retrieval-augmented generation (RAG) feedback.

The system jointly optimizes rewriting quality by leveraging:

📌 Retrieval effectiveness (MRR)
📌 Downstream QA performance (ROUGE)
📌 KL-regularized policy optimization
🚀 Overview

We train a rewriting model using online interaction with a retriever + generator pipeline, instead of static supervision.

Core idea:
User Query → Rewrite → Retrieve → Generate Answer → Compute Reward → PPO Update
🧩 Architecture
1. Policy Model (Rewrite Model)
Backbone: Flan-T5 (or other seq2seq models)
Output: rewritten query
2. Retriever
Dense retriever (e.g., ANCE / RoBERTa)
FAISS index for fast search
Metric: MRR
3. Generator (RAG QA Model)
LLM (e.g., LLaMA / other chat models)
Input: rewritten query + retrieved docs
Metric: ROUGE
4. Reward Function
Reward = MRR + λ * ROUGE - KL penalty

Where:

MRR → retrieval quality
ROUGE → answer quality
KL → deviation from reference policy
## Project Structure
.
├── online_ppo.py        # Main training script
├── datasets/            # Training data (TopiOCQA / QReCC)
├── index/               # FAISS index
├── checkpoints/         # Saved models
└── cache/               # HF dataset cache
## Installation
pip install torch transformers datasets trl faiss-cpu rouge-score pandas tqdm

## Dataset preparation
You could download our synthetic data from https://drive.google.com/drive/folders/15ZtUmV-kChi0ShvdgXzYOG6JRETycgOh?usp=drive_link

## Training
Before training, you need to carefully check the model/data paths in these files.
stage 1:
python train_t5_topiocqa.py
stage 2: 
bash train_online_ppo_topiocqa.sh \
  
