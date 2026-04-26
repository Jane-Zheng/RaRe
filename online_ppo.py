import os
import re
import json
import math
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_scheduler,
    AutoModelForSeq2SeqLM
    
)
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from rouge import Rouge
from tqdm import tqdm
import faiss
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

os.environ["WANDB_MODE"] = "offline"


# =========================
# online retriever globals
# =========================
RETRIEVER_MODEL = None
RETRIEVER_TOKENIZER = None
RETRIEVER_INDEX = None
RETRIEVER_INDEX_IDS = None
# =========================
# online generator globals
# =========================
GENERATOR_MODEL = None
GENERATOR_TOKENIZER = None
PID2DOC = None

def build_pid2doc(collection_tsv: str) -> Dict[Any, str]:
    """
    从 collection tsv 构建 pid -> passage/text 的映射
    优先用 passage，没有则退回 text
    """
    df = pd.read_csv(collection_tsv, sep="\t")
    text_col = "passage" if "passage" in df.columns else "text"
    df = df[df[text_col].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    pid2doc = {}
    for _, row in df.iterrows():
        pid2doc[row["id"]] = row[text_col]
    return pid2doc

def init_pid2doc(collection_tsv: str):
    global PID2DOC
    print("[Generator] Building pid2doc mapping...")
    PID2DOC = build_pid2doc(collection_tsv)
    print(f"[Generator] Loaded {len(PID2DOC)} passages")

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms

def get_topk_docs_from_pids(
    ranked_pid_lists: List[List[Any]],
    max_docs: int = 5,
) -> List[List[str]]:
    global PID2DOC
    if PID2DOC is None:
        raise RuntimeError("PID2DOC not initialized. Please call init_pid2doc().")

    batch_docs = []
    for pid_list in ranked_pid_lists:
        docs = []
        for pid in pid_list[:max_docs]:
            if pid in PID2DOC:
                docs.append(PID2DOC[pid])
        batch_docs.append(docs)
    return batch_docs

def load_collections(file_path: str):
    """
    只保留 passage 非空的文档
    你的 qrecc collection_tsv 用的是 passage 列
    """
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["passage"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    dataset = Dataset.from_pandas(df)
    return dataset


def get_embedding_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def read_index(index_path: str):
    index = faiss.read_index(index_path)
    return index


def get_embeddings_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(x) if x is not None else "" for x in texts[i:i + batch_size]]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            batch_embeddings = normalize_vectors(batch_embeddings)

        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def init_online_retriever(
    collection_tsv: str,
    embedding_model_path: str,
    index_path: str,
    device: str,
):
    """
    训练开始前调用一次
    """
    global RETRIEVER_MODEL, RETRIEVER_TOKENIZER, RETRIEVER_INDEX, RETRIEVER_INDEX_IDS

    print("[Retriever] Loading collection ids...")
    collection_dataset = load_collections(collection_tsv)
    RETRIEVER_INDEX_IDS = list(collection_dataset["id"])

    print("[Retriever] Loading embedding model...")
    RETRIEVER_MODEL, RETRIEVER_TOKENIZER = get_embedding_model(embedding_model_path, device)

    print("[Retriever] Loading FAISS index...")
    RETRIEVER_INDEX = read_index(index_path)

    print(f"[Retriever] Index size: {RETRIEVER_INDEX.ntotal}")
    print(f"[Retriever] Num ids: {len(RETRIEVER_INDEX_IDS)}")

def init_online_generator(
    generator_model_path: str,
    device: str,
):
    global GENERATOR_MODEL, GENERATOR_TOKENIZER

    print("[Generator] Loading tokenizer...")
    GENERATOR_TOKENIZER = AutoTokenizer.from_pretrained(generator_model_path, use_fast=True)
    GENERATOR_TOKENIZER.padding_side = "left"

    if GENERATOR_TOKENIZER.pad_token is None:
        GENERATOR_TOKENIZER.pad_token = GENERATOR_TOKENIZER.eos_token
        
    if GENERATOR_TOKENIZER.pad_token_id is None:
        GENERATOR_TOKENIZER.pad_token_id = GENERATOR_TOKENIZER.eos_token_id

    print(f"[Generator] Loading model on {device} ...")
    GENERATOR_MODEL = AutoModelForCausalLM.from_pretrained(
        generator_model_path,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map={"": int(device.split(":")[-1])},
    )
    GENERATOR_MODEL.eval()
    
    
def retrieve_topk_pids_for_queries(
    rewritten_queries: List[str],
    topk: int = 10,
    embedding_batch_size: int = 32,
) -> List[List[Any]]:
    """
    输入:
        rewritten_queries: ["rewrite1", "rewrite2", ...]
    输出:
        [
            [pid_1, pid_2, ...],   # 第1条rewrite的topk pid
            [pid_1, pid_2, ...],   # 第2条rewrite的topk pid
            ...
        ]
    """
    global RETRIEVER_MODEL, RETRIEVER_TOKENIZER, RETRIEVER_INDEX, RETRIEVER_INDEX_IDS

    if RETRIEVER_MODEL is None or RETRIEVER_TOKENIZER is None or RETRIEVER_INDEX is None or RETRIEVER_INDEX_IDS is None:
        raise RuntimeError("Retriever not initialized. Please call init_online_retriever() before training.")

    safe_queries = []
    for q in rewritten_queries:
        q = "" if q is None else str(q).strip()
        if len(q) == 0:
            q = " "
        safe_queries.append(q)

    query_embeddings = get_embeddings_batch(
        safe_queries,
        tokenizer=RETRIEVER_TOKENIZER,
        model=RETRIEVER_MODEL,
        device=next(RETRIEVER_MODEL.parameters()).device,
        batch_size=embedding_batch_size,
        max_length=512,
    )

    distances, indices = RETRIEVER_INDEX.search(query_embeddings.astype(np.float32), topk)

    ranked_pid_lists = []
    for row in indices:
        pid_list = []
        for doc_idx in row:
            if 0 <= doc_idx < len(RETRIEVER_INDEX_IDS):
                pid_list.append(RETRIEVER_INDEX_IDS[doc_idx])
        ranked_pid_lists.append(pid_list)

    return ranked_pid_lists

# =========================
# 1. utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_zscore_np(
    x: np.ndarray,
    eps: float = 1e-6,
    clip: float = 2.5,
    min_std: float = 1e-3,
) -> np.ndarray:
    """
    PPO-friendly zscore：
    - 避免 std 太小导致爆炸
    - 限制极值
    - 更稳定
    """
    if len(x) == 0:
        return x.astype(np.float32)

    x = x.astype(np.float32)

    mean = float(x.mean())
    std = float(x.std())

    # 👉 防止 std 太小放大噪声
    std = max(std, min_std)

    z = (x - mean) / (std + eps)

    # 👉 限制极端值（非常关键）
    if clip is not None:
        z = np.clip(z, -clip, clip)

    return z.astype(np.float32)

def zscore_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(x) == 0:
        return x
    mean = x.mean()
    std = x.std()
    if std < eps:
        return x - mean
    return (x - mean) / (std + eps)  # _before_0330_2


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# =========================
# 2. input / reward format
# =========================
def build_input(
    user_q,
    context: List[str],
    tokenizer,
    max_ctx_length: int = 512,
    max_suffix_length: int = 64,
) -> List[int]:
    """
    约束：
    - CTX 部分最多 512
    - 后缀 REWRITE: 单独放后面
    """
    rewrite_ids = tokenizer.encode(
        "REWRITE:",
        add_special_tokens=True,
        truncation=True,
        max_length=max_suffix_length,
    )
    query_ids = tokenizer.encode(
        "Query: " + user_q,
        add_special_tokens=True,
        truncation=True,
        max_length=128
    )

    input_ids = list(query_ids)

    history_ids = []
    for utt in reversed(context):
        utt_ids = tokenizer.encode(
            "CTX: " + utt + " <sep>",
            add_special_tokens=False,
            truncation=True,
            max_length=128,
        )

        if len(history_ids) + len(utt_ids) > max_ctx_length:
            remain = max_ctx_length - len(history_ids)
            if remain > 0:
                history_ids = utt_ids[-remain:] + history_ids
            break
        else:
            history_ids = utt_ids + history_ids

    input_ids = history_ids + rewrite_ids
    return input_ids

def build_input_qrecc(
    history,
    current_query,
    tokenizer,
    max_concat_length=512,
    max_query_length=64,
    per_turn_max_length=128
):
    # 1️⃣ 当前 query（放最前）
    query_ids = tokenizer.encode(
        "Query: " + current_query,
        add_special_tokens=True,
        truncation=True,
        max_length=max_query_length
    )

    input_ids = list(query_ids)

    # 2️⃣ 倒序拼接 context（最近优先）
    for utt in reversed(list(history)):
        # utt = 'Q: '+q+' A: '+ a
        utt_ids = tokenizer.encode(
            "CTX: " + utt + " <sep>",
            add_special_tokens=False,
            truncation=True,
            max_length=per_turn_max_length
        )

        if len(input_ids) + len(utt_ids) > max_concat_length:
            remain = max_concat_length - len(input_ids)
            if remain > 0:
                input_ids += utt_ids[:remain]
            break
        else:
            input_ids.extend(utt_ids)

    # 3️⃣ 加 REWRITE 触发生成
    rewrite_ids = tokenizer.encode(
        " REWRITE:",
        add_special_tokens=False
    )

    if len(input_ids) + len(rewrite_ids) <= max_concat_length:
        input_ids += rewrite_ids
    else:
        # 极端情况：强行留位置给 REWRITE
        input_ids = input_ids[:max_concat_length - len(rewrite_ids)] + rewrite_ids

    return input_ids
# =========================
# 3. dataset preprocess
# =========================
def filter_has_retrieval(example):
    return len(example.get("pos_docs_pids", [])) > 0


def make_preprocess_function_ppo(tokenizer, max_ctx_length=512, max_suffix_length=64):
    def preprocess_function_ppo(examples):
        input_ids = []
        attention_mask = []
        context_list = []
        rewrite_label_text = []
        answer_label_text = []
        pos_docs_pids = []
        cur_utt_text = []

        for user_q, context, rewrite_label, answer_label, pos_pids in zip(
            examples["cur_utt_text"],
            examples["ctx_utts_text"],
            examples["rewrite_res"],
            examples["cur_response_text"],          # 这里换成你的答案字段
            examples["pos_docs_pids"],
        ):
            # full_context = context 
            prompt_ids = build_input_qrecc(
                context,
                user_q,
                tokenizer=tokenizer,
                # max_ctx_length=max_ctx_length,
                # max_suffix_length=max_suffix_length,
            )

            input_ids.append(prompt_ids)
            attention_mask.append([1] * len(prompt_ids))
            context_list.append(context)
            rewrite_label_text.append(rewrite_label)
            answer_label_text.append(answer_label)
            pos_docs_pids.append(pos_pids)
            cur_utt_text.append(user_q)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_list": context_list,
            "rewrite_label_text": rewrite_label_text,
            "answer_label_text": answer_label_text,
            "pos_docs_pids": pos_docs_pids,
            "cur_utt_text": cur_utt_text,
        }

    return preprocess_function_ppo

def ppo_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {
        "input_ids": [
            torch.tensor(f["input_ids"], dtype=torch.long) for f in features
        ],
        "attention_mask": [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
        ],
        "context_list": [f["context_list"] for f in features],
        "rewrite_label_text": [f["rewrite_label_text"] for f in features],
        "answer_label_text": [f["answer_label_text"] for f in features],
        "pos_docs_pids": [f["pos_docs_pids"] for f in features],
        "cur_utt_text": [f["cur_utt_text"] for f in features],
    }
    return batch

def build_rag_prompt(query: str, docs: List[str]) -> str: # 0330修正
    # joined_docs = "\n\n".join([f"[Doc {i+1}] {d}" for i, d in enumerate(docs)])
    joined_docs = "\n\n".join(docs) if len(docs) > 0 else ""
    
    # doc = "\n\n".join(docs[0])
    # doc = docs[0] if len(docs) > 0 else ""
    #  prompt = (
    #     "You are a helpful QA assistant.\n"
    #     "Answer the user question based on the retrieved documents.\n"
    #     "If the answer is not supported by the documents, say you are not sure.\n\n"
    #     f"Question: {query}\n\n"
    #     f"Retrieved Documents:\n{joined_docs}\n\n"
    #     "Answer:"
    # )
    
    system_prompt = """You are a factual and very smart assistant. Understand the provided knowledge and Strictly answer the query based on the provided knowledge.
	                       Rules:
	                         1. Keep answers precise and short only include the key words or conclusion.
	                         2. Quote directly from the knowledge when possible.
                         """
    rag_message = create_message(query, joined_docs)  
    messages = [
        {"role": "system", "content": system_prompt},
        rag_message,
    ]
       
    return messages

def create_message(query, doc):
    # topiocqa
    # answer_inputs = 'Query:{}, Knowledge:{}. Answer the query strictly based on the knowledge above in 1-3 phrases if you can; else, it should be in 1-2 sentences. If unsure, say "UNANSWERABLE" only.'.format(query, doc)
    # for qrecc
    answer_inputs = 'Query:{}, Knowledge:{}. Answer the query strictly in 1-3 phrases.'.format(query, doc)
    answer_message = {'role':'user', 'content': answer_inputs}
    return answer_message

def clean_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'^(assistant|Assistant)\s*[:：]?\s*', '', ans)
    ans = re.sub(r'^(Answer|Final Answer)\s*[:：]\s*', '', ans, flags=re.I)
    ans = re.sub(r'Answer the query strictly in 1-3 phrases\.?', '', ans, flags=re.I)
    return ans.strip()

def generate_answers_from_rewrites(
    rewritten_queries: List[str],
    ranked_pid_lists: List[List[Any]],
    max_docs: int = 5,
    max_new_tokens: int = 128,
    gen_batch_size: int = 4,
) -> List[str]:
    global GENERATOR_MODEL, GENERATOR_TOKENIZER

    if GENERATOR_MODEL is None or GENERATOR_TOKENIZER is None:
        raise RuntimeError("Generator not initialized.")

    batch_docs = get_topk_docs_from_pids(ranked_pid_lists, max_docs=max_docs)
    prompts = [build_rag_prompt(q, docs) for q, docs in zip(rewritten_queries, batch_docs)]

    outputs_all = []
    for i in range(0, len(prompts), gen_batch_size):
        sub_prompts = prompts[i:i + gen_batch_size]

        enc = GENERATOR_TOKENIZER.apply_chat_template(
            sub_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(GENERATOR_MODEL.device)

        with torch.no_grad():
            out = GENERATOR_MODEL.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=GENERATOR_TOKENIZER.pad_token_id,
                eos_token_id=GENERATOR_TOKENIZER.eos_token_id,
            )

        # input_lens = enc["attention_mask"].sum(dim=1)

        # for j in range(out.size(0)):
        #     gen_ids = out[j, input_lens[j]:]
        #     ans = GENERATOR_TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()
        #     outputs_all.append(ans)
            
        prompt_len = enc["input_ids"].shape[1] # 0401修正

        for j in range(out.size(0)):
            gen_ids = out[j, prompt_len:]
            ans = GENERATOR_TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()
            ans = clean_answer(ans)
            outputs_all.append(ans)

    return outputs_all

# =========================
# 4. online reward
# =========================

def compute_mrr_from_ranked_pids(
    ranked_pids: List[Any],
    gold_pids: List[Any],
) -> float:
    """
    标准 MRR：第一个命中相关文档的位置倒数
    """
    if not ranked_pids or not gold_pids:
        return 0.0

    gold_set = set(gold_pids)
    for rank, pid in enumerate(ranked_pids, start=1):
        if pid in gold_set:
            return 1.0 / rank
    return 0.0

def score_retrieval_mrr_batch(
    batch_res: List[str],
    batch_gold_pids: List[List[Any]],
    retriever_topk: int = 10,
    retriever_embed_batch_size: int = 32,
) -> Tuple[List[float], List[List[Any]]]:
    ranked_pid_lists = retrieve_topk_pids_for_queries(
        rewritten_queries=batch_res,
        topk=retriever_topk,
        embedding_batch_size=retriever_embed_batch_size,
    )

    mrr_scores = []
    for ranked_pids, gold_pids in zip(ranked_pid_lists, batch_gold_pids):
        mrr = compute_mrr_from_ranked_pids(ranked_pids, gold_pids)
        mrr_scores.append(mrr)

    return mrr_scores, ranked_pid_lists

def expand_batch_for_multi_sample(batch: Dict[str, Any], num_samples_per_query: int) -> Dict[str, Any]:
    if num_samples_per_query <= 1:
        return batch

    expanded = {
        "input_ids": [],
        "attention_mask": [],
        "context_list": [],
        "rewrite_label_text": [],
        "answer_label_text": [],
        "pos_docs_pids": [],
        "cur_utt_text": [],
    }

    for i in range(len(batch["input_ids"])):
        for _ in range(num_samples_per_query):
            expanded["input_ids"].append(batch["input_ids"][i].clone())
            expanded["attention_mask"].append(batch["attention_mask"][i].clone())
            expanded["context_list"].append(batch["context_list"][i])
            expanded["rewrite_label_text"].append(batch["rewrite_label_text"][i])
            expanded["answer_label_text"].append(batch["answer_label_text"][i])
            expanded["pos_docs_pids"].append(batch["pos_docs_pids"][i])
            expanded["cur_utt_text"].append(batch["cur_utt_text"][i])

    return expanded

def score_generation_rouge_batch(
    generated_answers: List[str],
    batch_answer_label: List[str],
) -> List[float]:
    """
    在线 generation reward：
    rewrite -> retrieve -> downstream generator answer -> ROUGE-L(answer, gold_answer)
    """
    rouge_metric = Rouge(metrics=["rouge-1"], stats=["f"]) # rouge-l

    safe_res = [x if isinstance(x, str) and len(x.strip()) > 0 else " " for x in generated_answers]
    safe_label = [x if isinstance(x, str) and len(x.strip()) > 0 else " " for x in batch_answer_label]

    try:
        metric_dict = rouge_metric.get_scores(
            safe_res, safe_label, avg=False, ignore_empty=True
        )
        scores = [safe_float(x["rouge-1"]["f"], 0.0) for x in metric_dict] # rouge-l
    except Exception:
        scores = [0.0 for _ in generated_answers]
    return scores

def combine_rewards(
    mrr_scores: List[float],
    rouge_scores: List[float],
    mrr_weight: float = 1.0,
    rouge_weight: float = 0.0,
    rouge_threshold: Optional[float] = None,
    rouge_penalty: float = 0.0,
    normalize_mrr: bool = True,
    normalize_rouge: bool = True,
) -> Dict[str, List[float]]:
    """
    推荐默认：
    - MRR 主导
    - ROUGE 只做小权重辅助，或者阈值惩罚
    """
    mrr_np = np.array(mrr_scores, dtype=np.float32)
    rouge_np = np.array(rouge_scores, dtype=np.float32)

    if normalize_mrr:
        mrr_used = zscore_np(mrr_np)
    else:
        mrr_used = mrr_np

    if normalize_rouge:
        rouge_used = zscore_np(rouge_np)
    else:
        rouge_used = rouge_np

    total = mrr_weight * mrr_used + rouge_weight * rouge_used
    total = np.tanh(total) # 
    # total = np.log(1 + total) # 忘记注释掉了。。。。。

    if rouge_threshold is not None and rouge_penalty > 0.0:
        penalty_mask = rouge_np < rouge_threshold
        total = total - rouge_penalty * penalty_mask.astype(np.float32)

    return {
        "mrr_raw": mrr_np.tolist(),
        "rouge_raw": rouge_np.tolist(),
        "mrr_used": mrr_used.tolist(),
        "rouge_used": rouge_used.tolist(),
        "total_reward": total.tolist(),
    } # old version before 0330


def minmax_np(
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    映射到 [0, 1]，适合原始指标本身有明确上下界的情况。
    """
    x = x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min + eps)).astype(np.float32)


def squash_reward(
    x: np.ndarray,
    method: str = "tanh",
    scale: float = 1.0,
    clip_value: Optional[float] = None,
) -> np.ndarray:
    """
    对合成后的 reward 做压缩，减小方差。
    """
    x = x.astype(np.float32)

    if method == "none":
        y = x
    elif method == "tanh":
        y = np.tanh(x / max(scale, 1e-6))
    elif method == "clip":
        if clip_value is None:
            clip_value = 1.0
        y = np.clip(x, -clip_value, clip_value)
    else:
        raise ValueError(f"Unsupported squash method: {method}")

    return y.astype(np.float32)

# new soft rewards after 0330-2
# def combine_rewards(
#     mrr_scores: List[float],
#     rouge_scores: List[float],
#     mrr_weight: float = 1.0,
#     rouge_weight: float = 0.0,
#     rouge_threshold: Optional[float] = None,
#     rouge_penalty: float = 0.0,
#     normalize_mrr: bool = True,
#     normalize_rouge: bool = True,
#     normalize_method: str = "zscore",   # "zscore" | "minmax" | "none"
#     zscore_clip: float = 2.5,
#     final_squash: str = "tanh",         # "tanh" | "clip" | "none"
#     final_scale: float = 1.5,
#     final_clip_value: float = 1.0,
#     center_final_reward: bool = False,
# ) -> Dict[str, List[float]]:
#     """
#     更适合 PPO 的 reward 合成版本。

#     设计目标：
#     - MRR 主导
#     - ROUGE 作为辅助
#     - 降低 reward 离散性和极值影响
#     - 让最终 reward 更平滑、更稳定

#     建议默认：
#     - mrr_weight=1.0
#     - rouge_weight=0.1~0.3
#     - normalize_method="zscore"
#     - final_squash="tanh"
#     """

#     mrr_np = np.asarray(mrr_scores, dtype=np.float32)
#     rouge_np = np.asarray(rouge_scores, dtype=np.float32)

#     if len(mrr_np) != len(rouge_np):
#         raise ValueError("mrr_scores and rouge_scores must have the same length")

#     if normalize_method not in {"zscore", "minmax", "none"}:
#         raise ValueError("normalize_method must be one of: 'zscore', 'minmax', 'none'")

#     def _normalize(x: np.ndarray, use_norm: bool) -> np.ndarray:
#         if not use_norm or normalize_method == "none":
#             return x.astype(np.float32)

#         if normalize_method == "zscore":
#             return safe_zscore_np(x, clip=zscore_clip)

#         if normalize_method == "minmax":
#             return minmax_np(x)

#         raise ValueError("Unexpected normalize_method")

#     mrr_used = _normalize(mrr_np, normalize_mrr)
#     rouge_used = _normalize(rouge_np, normalize_rouge)

#     # 基础 reward
#     base_total = (mrr_weight * mrr_used + rouge_weight * rouge_used).astype(np.float32)

#     # 阈值惩罚
#     # 这里仍保留硬阈值，但你可以把 penalty 设小一点，避免 reward 跳变太大
#     penalty = np.zeros_like(base_total, dtype=np.float32)
#     if rouge_threshold is not None and rouge_penalty > 0.0:
#         # penalty_mask = rouge_np < rouge_threshold
#         # penalty = rouge_penalty * penalty_mask.astype(np.float32)
#         penalty = rouge_penalty * np.maximum(0.0, rouge_threshold - rouge_np).astype(np.float32)
       

#     total_before_squash = base_total - penalty

#     # 最终压缩，降低极值和离散跳跃对 PPO 的冲击
#     total_reward = squash_reward(
#         total_before_squash,
#         method=final_squash,
#         scale=final_scale,
#         clip_value=final_clip_value,
#     )

#     # 可选：再做一次中心化，但不建议每次都开
#     # 因为有些 PPO 实现里 advantage 本身还会再 normalize
#     if center_final_reward:
#         total_reward = total_reward - np.mean(total_reward, dtype=np.float32)

#     return {
#         "mrr_raw": mrr_np.tolist(),
#         "rouge_raw": rouge_np.tolist(),
#         "mrr_used": mrr_used.tolist(),
#         "rouge_used": rouge_used.tolist(),
#         "base_total": base_total.tolist(),
#         "penalty": penalty.tolist(),
#         "total_before_squash": total_before_squash.tolist(),
#         "total_reward": total_reward.tolist(),
#     }


def get_online_rewards(
    batch_res: List[str],
    batch_answer_label: List[str],
    batch_gold_pids: List[List[Any]],
    retriever_topk: int,
    retriever_embed_batch_size: int,
    generator_max_docs: int,
    generator_max_new_tokens: int,
    generator_batch_size: int,
    mrr_weight: float,
    rouge_weight: float,
    rouge_threshold: Optional[float],
    rouge_penalty: float,
    normalize_mrr: bool,
    normalize_rouge: bool,
    enable_generation_reward: bool,
):
    mrr_scores, ranked_pid_lists = score_retrieval_mrr_batch(
        batch_res=batch_res,
        batch_gold_pids=batch_gold_pids,
        retriever_topk=retriever_topk,
        retriever_embed_batch_size=retriever_embed_batch_size,
    )

    generated_answers = None
    if enable_generation_reward:
        generated_answers = generate_answers_from_rewrites(
            rewritten_queries=batch_res,
            ranked_pid_lists=ranked_pid_lists,
            max_docs=generator_max_docs,
            max_new_tokens=generator_max_new_tokens,
            gen_batch_size=generator_batch_size,
        )

        rouge_scores = score_generation_rouge_batch(
            generated_answers=generated_answers,
            batch_answer_label=batch_answer_label,
        )
    else:
        rouge_scores = [0.0 for _ in batch_res]

    reward_dict = combine_rewards(
        mrr_scores=mrr_scores,
        rouge_scores=rouge_scores,
        mrr_weight=mrr_weight,
        rouge_weight=rouge_weight,
        rouge_threshold=rouge_threshold,
        rouge_penalty=rouge_penalty,
        normalize_mrr=normalize_mrr,
        normalize_rouge=normalize_rouge,
    )

    rewards = [
        torch.tensor(float(x), dtype=torch.float32)
        for x in reward_dict["total_reward"]
    ]

    reward_dict["generated_answers"] = (
        generated_answers if generated_answers is not None else [""] * len(batch_res)
    )
    return rewards, reward_dict, ranked_pid_lists


# =========================
# 5. args
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42) 

    parser.add_argument("--train_file", type=str, default="./project/RAG-test/qrecc/qrecc_train.jsonl")
    parser.add_argument("--val_file", type=str, default="./project/RAG-test/qrecc/qrecc_val.jsonl")
    parser.add_argument("--cache_dir", type=str, default="./project/RAG-test/RL/PPO/cache")

    parser.add_argument("--base_model_name", type=str, default="./project/RAG-test/Flan-T5-large")
    parser.add_argument("--policy_model_name", type=str, default="./project/RAG-test/t5-checkpoint/t5-b1g2-qrecc-1024-256-5e-4-e5-full-inst-0110/checkpoint-70700")

    parser.add_argument("--output_dir", type=str, default="./project/RAG-test/RL/PPO/checkpoint/online_ppo_qrecc")
    parser.add_argument("--logging_dir", type=str, default="./project/RAG-test/RL/PPO/checkpoint/online_ppo_qrecc/logs")

    parser.add_argument("--max_ctx_length", type=int, default=512)
    parser.add_argument("--max_suffix_length", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--ppo_batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=50)

    # online reward
    parser.add_argument("--retriever_topk", type=int, default=10)
    parser.add_argument("--mrr_weight", type=float, default=1.0)
    parser.add_argument("--rouge_weight", type=float, default=0.1)
    parser.add_argument("--rouge_threshold", type=float, default=None)
    parser.add_argument("--rouge_penalty", type=float, default=0.0)
    parser.add_argument("--normalize_mrr", action="store_true")
    parser.add_argument("--normalize_rouge", action="store_true")
    parser.add_argument("--force_enable_generator", action="store_true")
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument(
    "--retriever_collection_tsv",
    type=str,
    default="./project/Llama2/datasets/qrecc/selected_val_qrecc_segments.tsv",
    )
    parser.add_argument(
    "--retriever_model_path",
    type=str,
    default="./project/RAG-test/msmarco-roberta-base-ance-firstp",
    )
    parser.add_argument(
    "--retriever_index_path",
    type=str,
    default="./project/RAG-test/first_part_test/qrecc_index/select_test_qrecc_faiss_cos.index",
    )
    parser.add_argument("--retriever_embed_batch_size", type=int, default=32)
    
    parser.add_argument("--num_samples_per_query", type=int, default=2)

    parser.add_argument(
    "--generator_model_path",
    type=str,
    default="/path/to/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--generator_max_docs", type=int, default=5)
    parser.add_argument("--generator_max_new_tokens", type=int, default=128)
    parser.add_argument("--generator_batch_size", type=int, default=4)
    
    # device
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--reward_device", type=str, default="cuda:1")

    # sample
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    return args

def need_generation_reward(args) -> bool:
    return args.force_enable_generator or (
        args.rouge_weight > 0.0
        or (args.rouge_threshold is not None and args.rouge_penalty > 0.0)
    )
    

def pad_encoder_batch(
    input_tensors: List[torch.Tensor],
    pad_token_id: int,
    device: str,
):
    input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=pad_token_id).to(device)
    attention_mask = (input_ids != pad_token_id).long()
    return input_ids, attention_mask


def strip_decoder_start_token(
    sequences: torch.Tensor,
    decoder_start_token_id: Optional[int],
):
    """
    T5/Flan-T5 generate 常常在最前面带一个 decoder_start_token_id（通常是 0）
    PPO 的 action 我们只保留真正生成的 token。
    """
    if sequences.size(1) == 0:
        return sequences
    if decoder_start_token_id is None:
        return sequences
    if torch.all(sequences[:, 0] == decoder_start_token_id):
        return sequences[:, 1:]
    return sequences


def get_model_outputs_and_values(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
):
    """
    对 sampled response 做 teacher-forcing，得到 token-level logits 和 values
    """
    # T5 shift-right
    decoder_input_ids = model.pretrained_model._shift_right(response_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

    # 兼容不同 trl 版本的返回格式
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs[0]

    if hasattr(outputs, "value"):
        values = outputs.value
    elif hasattr(outputs, "values"):
        values = outputs.values
    else:
        values = outputs[-1]

    return logits, values


def sequence_logprobs_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int,
):
    """
    logits: [B, T, V]
    target_ids: [B, T]
    返回:
      token_logprobs: [B, T]
      seq_logprobs: [B]
      mask: [B, T]
    """
    logprobs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(logprobs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != pad_token_id).float()
    token_logprobs = gathered * mask
    lengths = mask.sum(dim=-1).clamp(min=1.0)
    seq_logprobs = token_logprobs.sum(dim=-1) / lengths
    return token_logprobs, seq_logprobs, mask


def get_sequence_values(
    values: torch.Tensor,
    mask: torch.Tensor,
):
    """
    values: [B, T]
    mask: [B, T]
    取最后一个有效 token 位置的 value 作为序列 value
    """
    lengths = mask.sum(dim=-1).long().clamp(min=1) - 1
    batch_idx = torch.arange(values.size(0), device=values.device)
    seq_values = values[batch_idx, lengths]
    return seq_values


def whiten(x: torch.Tensor, eps: float = 1e-8):
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / (std + eps)


def compute_kl_penalty(
    policy_seq_logprobs: torch.Tensor,
    ref_seq_logprobs: torch.Tensor,
):
    """
    一个简单的 sequence-level KL proxy
    """
    return policy_seq_logprobs - ref_seq_logprobs

def compute_ref_penalty(old_token_logprobs, ref_token_logprobs, mask):
    diff = torch.abs(old_token_logprobs - ref_token_logprobs) * mask
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return diff.sum(dim=-1) / denom

def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
    normalize: bool = True,
):
    """
    rewards: [batch_size * group_size]
    returns: advantages with same shape
    """
    assert rewards.dim() == 1
    assert rewards.size(0) % group_size == 0

    num_groups = rewards.size(0) // group_size
    grouped = rewards.view(num_groups, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)
    if normalize:
        group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = (grouped - group_mean) / (group_std + eps)
    else:
        advantages = grouped - group_mean

    return advantages.view(-1)

def ppo_update(
    model,
    optimizer,
    scheduler,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    old_seq_logprobs: torch.Tensor,
    old_seq_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    pad_token_id: int,
    cliprange: float,
    cliprange_value: float,
    vf_coef: float,
    ppo_epochs: int,
    mini_batch_size: int,
):
    model.train()
    batch_size = input_ids.size(0)

    stats = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clipfrac": [],
    }

    for _ in range(ppo_epochs):
        perm = torch.randperm(batch_size, device=input_ids.device)

        for start in range(0, batch_size, mini_batch_size):
            idx = perm[start:start + mini_batch_size]

            mb_input_ids = input_ids[idx]
            mb_attention_mask = attention_mask[idx]
            mb_response_ids = response_ids[idx]
            mb_old_logprobs = old_seq_logprobs[idx]
            mb_old_values = old_seq_values[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            logits, values = get_model_outputs_and_values(
                model,
                mb_input_ids,
                mb_attention_mask,
                mb_response_ids,
            )

            _, seq_logprobs, mask = sequence_logprobs_from_logits(
                logits, mb_response_ids, pad_token_id
            )
            seq_values = get_sequence_values(values, mask)

            logratio = seq_logprobs - mb_old_logprobs
            ratio = torch.exp(logratio)

            pg_loss_1 = -mb_advantages * ratio
            pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            value_pred_clipped = mb_old_values + torch.clamp(
                seq_values - mb_old_values,
                -cliprange_value,
                cliprange_value,
            )
            value_loss_1 = (seq_values - mb_returns) ** 2
            value_loss_2 = (value_pred_clipped - mb_returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

            entropy = -(torch.exp(F.log_softmax(logits, dim=-1)) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            entropy = (entropy * mask).sum() / mask.sum().clamp(min=1.0)

            loss = policy_loss + vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            approx_kl = 0.5 * ((seq_logprobs - mb_old_logprobs) ** 2).mean()
            clipfrac = ((ratio > 1.0 + cliprange) | (ratio < 1.0 - cliprange)).float().mean()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl.item())
            stats["clipfrac"].append(clipfrac.item())

    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in stats.items()}

# =========================
# 6. main
# =========================
def main():
    args = get_args()
    set_seed(args.seed)

    # print(f"device = {args.device}")
    enable_generation_reward = need_generation_reward(args)
    print(f"enable_generation_reward = {enable_generation_reward}")

    # ---------- tokenizer ----------
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    num_added = 0
    if tokenizer.sep_token is None or tokenizer.sep_token != "<sep>":
        num_added = tokenizer.add_special_tokens({"sep_token": "<sep>"})

    # ---------- model ----------
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        args.policy_model_name,
        torch_dtype=torch_dtype,
        device_map={"": 1},
    )

    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        args.policy_model_name,
        torch_dtype=torch_dtype,
        device_map={"": 1},
    )

    if num_added > 0:
        policy_model.pretrained_model.resize_token_embeddings(len(tokenizer))
        ref_model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ref_model.v_head.load_state_dict(policy_model.v_head.state_dict())
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    policy_model.train()
    
    # ---------- init online retriever ----------
    init_online_retriever(
    collection_tsv=args.retriever_collection_tsv,
    embedding_model_path=args.retriever_model_path,
    index_path=args.retriever_index_path,
    device=args.reward_device,
    )
    # ---------- init online generator ----------
    if need_generation_reward(args):
        init_pid2doc(args.retriever_collection_tsv)
        init_online_generator(
        generator_model_path=args.generator_model_path,
        device=args.reward_device,
        )
    else:
        print("[Generator] Disabled because rouge reward is not used.")

    # ---------- dataset ----------
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
        cache_dir=args.cache_dir,
        num_proc=8,
    )

    dataset = dataset.filter(filter_has_retrieval, num_proc=8)

    preprocess_function_ppo = make_preprocess_function_ppo(
        tokenizer=tokenizer,
        max_ctx_length=args.max_ctx_length,
        max_suffix_length=args.max_suffix_length,
    )

    processed_dataset = dataset.map(
        preprocess_function_ppo,
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names,
        num_proc=16,
        load_from_cache_file=True,
    )

    print(processed_dataset)

    # ---------- optimizer / scheduler ----------
    optimizer = optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    num_minibatches = max(1, (args.ppo_batch_size * args.num_samples_per_query) // args.mini_batch_size)
    approx_optimizer_steps = args.total_steps * args.ppo_epochs * num_minibatches

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=approx_optimizer_steps,
    )
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_length": -1,
        "do_sample": True,
        "top_k": 0,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    
    train_dataloader = DataLoader(
        processed_dataset["train"],
        batch_size=args.ppo_batch_size,
        shuffle=True,
        collate_fn=ppo_collator,
    )

    policy_model.train()
    ref_model.eval()

    print(
        {
            "mrr_weight": args.mrr_weight,
            "rouge_weight": args.rouge_weight,
            "rouge_threshold": args.rouge_threshold,
            "rouge_penalty": args.rouge_penalty,
            "normalize_mrr": args.normalize_mrr,
            "normalize_rouge": args.normalize_rouge,
            "kl_coef": args.kl_coef,
            "cliprange": args.cliprange,
            "cliprange_value": args.cliprange_value,
            "vf_coef": args.vf_coef,
        }
    )

    global_step = 0
    epoch = 0

    decoder_start_token_id = policy_model.pretrained_model.config.decoder_start_token_id
    pad_token_id = tokenizer.pad_token_id

    while global_step < args.total_steps:
        print(f"\n===== Starting epoch {epoch} =====")

        for batch in tqdm(train_dataloader, desc=f"ppo-train-epoch-{epoch}"):
            if global_step >= args.total_steps:
                break

            # ===== multi-sample expand =====
            batch = expand_batch_for_multi_sample(batch, args.num_samples_per_query)

            # ===== encoder batch pad =====
            input_ids, attention_mask = pad_encoder_batch(
                batch["input_ids"],
                pad_token_id=pad_token_id,
                device=args.policy_device,
            )

            # ===== rollout with current policy =====
            with torch.no_grad():
                gen_outputs = policy_model.pretrained_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_k=0,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                response_ids = strip_decoder_start_token(
                    gen_outputs.sequences,
                    decoder_start_token_id=decoder_start_token_id,
                )

                ref_gen_outputs = ref_model.pretrained_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_k=0,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                ref_response_ids = strip_decoder_start_token(
                    ref_gen_outputs.sequences,
                    decoder_start_token_id=decoder_start_token_id,
                )

            batch_response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            batch_ref_response = tokenizer.batch_decode(ref_response_ids, skip_special_tokens=True)

            if len(batch_response) == 0:
                continue

            # ===== online env reward =====
            rewards, reward_dict, ranked_pid_lists = get_online_rewards(
                batch_res=batch_response,
                batch_answer_label=batch["answer_label_text"],
                batch_gold_pids=batch["pos_docs_pids"],
                retriever_topk=args.retriever_topk,
                retriever_embed_batch_size=args.retriever_embed_batch_size,
                generator_max_docs=args.generator_max_docs,
                generator_max_new_tokens=args.generator_max_new_tokens,
                generator_batch_size=args.generator_batch_size,
                mrr_weight=args.mrr_weight,
                rouge_weight=args.rouge_weight,
                rouge_threshold=args.rouge_threshold,
                rouge_penalty=args.rouge_penalty,
                normalize_mrr=args.normalize_mrr,
                normalize_rouge=args.normalize_rouge,
                enable_generation_reward=enable_generation_reward,
            )

            ref_rewards, ref_reward_dict, _ = get_online_rewards(
                batch_res=batch_ref_response,
                batch_answer_label=batch["answer_label_text"],
                batch_gold_pids=batch["pos_docs_pids"],
                retriever_topk=args.retriever_topk,
                retriever_embed_batch_size=args.retriever_embed_batch_size,
                generator_max_docs=args.generator_max_docs,
                generator_max_new_tokens=args.generator_max_new_tokens,
                generator_batch_size=args.generator_batch_size,
                mrr_weight=args.mrr_weight,
                rouge_weight=args.rouge_weight,
                rouge_threshold=args.rouge_threshold,
                rouge_penalty=args.rouge_penalty,
                normalize_mrr=args.normalize_mrr,
                normalize_rouge=args.normalize_rouge,
                enable_generation_reward=enable_generation_reward,
            )

            reward_tensor = torch.tensor(
                # [float(x.item()) for x in rewards],
                [float(r.item() - rr.item()) for r, rr in zip(rewards, ref_rewards)],
                dtype=torch.float32,
                device=args.policy_device,
            )

            # ===== old policy logprob/value =====
            response_ids = response_ids.to(args.policy_device)
            with torch.no_grad():
                old_logits, old_values = get_model_outputs_and_values(
                    policy_model,
                    input_ids,
                    attention_mask,
                    response_ids,
                )
                old_token_logprobs, old_seq_logprobs, mask = sequence_logprobs_from_logits(
                    old_logits, response_ids, pad_token_id
                )
                old_seq_values = get_sequence_values(old_values, mask)

                ref_logits, _ = get_model_outputs_and_values(
                    ref_model,
                    input_ids,
                    attention_mask,
                    response_ids,
                )
                ref_token_logprobs, ref_seq_logprobs, _ = sequence_logprobs_from_logits(
                    ref_logits, response_ids, pad_token_id
                )

            # ===== KL-adjusted reward =====
            # kl = compute_kl_penalty(old_seq_logprobs, ref_seq_logprobs) # 0327_2
            kl = compute_ref_penalty(old_token_logprobs, ref_token_logprobs,mask) # 0328_1
            # kl_penalty = torch.clamp(kl, min=0.0)
            kl_penalty = torch.abs(kl)
            total_rewards = reward_tensor - args.kl_coef * kl_penalty

            # ===== advantage / return =====
            advantages = total_rewards - old_seq_values
            
            # advantages = compute_grpo_advantages(
            #     total_rewards,
            #     group_size=args.num_samples_per_query,
            # ) # 0329_2
            # advantages = whiten(advantages) # ?0330 说是要注释掉？？grpo的测试耗时太可怕了。。。要不还是改回0328_1的设置？
            # returns = advantages + old_seq_values
            returns = total_rewards # 0328_1

            # ===== PPO update =====
            stats = ppo_update(
                model=policy_model,
                optimizer=optimizer,
                scheduler=scheduler,
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_ids=response_ids,
                old_seq_logprobs=old_seq_logprobs.detach(),
                old_seq_values=old_seq_values.detach(),
                advantages=advantages.detach(),
                returns=returns.detach(),
                pad_token_id=pad_token_id,
                cliprange=args.cliprange,
                cliprange_value=args.cliprange_value,
                vf_coef=args.vf_coef,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.mini_batch_size,
            )

            if global_step % args.log_freq == 0:
                print("=" * 120)
                print(f"epoch = {epoch}, step = {global_step}")
                print(f"context = {' || '.join(batch['context_list'][0])}")
                print(f"rewrite = {batch_response[0]}")
                print(f"ref_rewrite = {batch_ref_response[0]}")
                print(f"generated_answer = {reward_dict['generated_answers'][0]}")
                print(f"gold_answer = {batch['answer_label_text'][0]}")
                print(f"mrr_raw = {reward_dict['mrr_raw'][0]:.6f}")
                print(f"rouge_raw = {reward_dict['rouge_raw'][0]:.6f}")
                print(f"env_reward = {float(reward_tensor[0].item()):.6f}")
                print(f"kl = {float(kl[0].item()):.6f}")
                print(f"total_reward = {float(total_rewards[0].item()):.6f}")
                print(f"ref_env_reward = {float(ref_rewards[0].item()):.6f}")
                print(f"policy_loss = {stats['policy_loss']:.6f}")
                print(f"value_loss = {stats['value_loss']:.6f}")
                print(f"approx_kl = {stats['approx_kl']:.6f}")
                print(f"clipfrac = {stats['clipfrac']:.6f}")
                print(f"ranked_pids = {ranked_pid_lists[0][:5] if len(ranked_pid_lists[0]) > 5 else ranked_pid_lists[0]}")

            global_step += 1

            if global_step > 0 and global_step % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, f"step_{global_step}")
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"saved to {save_path}")

            if global_step >= args.total_steps:
                break

        epoch += 1

    final_path = os.path.join(args.output_dir, "step_final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"final saved to {final_path}")

if __name__ == "__main__":
    main()
