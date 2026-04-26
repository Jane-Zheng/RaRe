import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    Seq2SeqTrainer, Seq2SeqTrainingArguments, 
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
)
import numpy as np
from evaluate import load

# version 2 no "rewrite"
# def build_input_topiocqa(
#     history_queries, history_answers, current_query, tokenizer,
#     max_concat_length=512, max_query_length=64, per_turn_max_length=128
# ):
#     current_ids = tokenizer.encode(
#         "question: " + current_query,
#         add_special_tokens=True,
#         truncation=True,
#         max_length=max_query_length
#     )

#     input_ids = list(current_ids)

#     first_context = True
#     for q, a in reversed(list(zip(history_queries, history_answers))):
#         turn_text = f"{q} {a}"
#         if first_context:
#             turn_text = "context: " + turn_text
#             first_context = False

#         turn_ids = tokenizer.encode(
#             turn_text,
#             add_special_tokens=True,
#             truncation=True,
#             max_length=per_turn_max_length
#         )

#         if len(input_ids) + len(turn_ids) > max_concat_length:
#             remain = max_concat_length - len(input_ids)
#             if remain > 1:
#                 input_ids += turn_ids[:remain - 1] + [turn_ids[-1]]
#             break
#         else:
#             input_ids.extend(turn_ids)

#     return input_ids

# seems not good
# def build_input_topiocqa(history_queries, history_answers, current_query, tokenizer,
#                          max_concat_length=512, max_query_length=64, per_turn_max_length=128):
#     """
#     TopiOCQA:
#     - 历史按 QA turn 成对处理
#     - 最近 turn 优先（倒序截断）
#     - 每个 turn 内保持 Q -> A
#     """

#     # 当前 query + rewrite 提示
#     rewrite_ids = tokenizer.encode(
#         "CURRENT_Q: " + current_query + " REWRITE:",
#         add_special_tokens=True,
#         truncation=True,
#         max_length=max_query_length
#     )

#     history_ids = []

#     # 倒序遍历最近历史 turn
#     for q, a in reversed(list(zip(history_queries, history_answers))):
#         turn_text = f"Q: {q} A: {a} <sep>"
#         turn_ids = tokenizer.encode(
#             turn_text,
#             add_special_tokens=False,
#             truncation=True,
#             max_length=per_turn_max_length
#         )

#         if len(history_ids) + len(turn_ids) > max_concat_length:
#             remain = max_concat_length - len(history_ids)
#             if remain > 0:
#                 # 这里保留 turn 的尾部并不完美，但至少最近信息优先
#                 history_ids = turn_ids[-remain:] + history_ids
#             break
#         else:
#             history_ids = turn_ids + history_ids

#     input_ids = history_ids + rewrite_ids
#     return input_ids

def build_input_topiocqa(
    history_queries,
    history_answers,
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
    for q, a in reversed(list(zip(history_queries, history_answers))):
        utt = 'Q: '+q+' A: '+ a
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

# ver qrecc-like
def preprocess_function_topiocqa(examples, tokenizer,
                                 max_concat_length=512,
                                 max_query_length=64,
                                 per_turn_max_length=128,
                                 target_max_length=64):
    input_ids = []
    attention_mask = []
    targets = []

    for history_queries, history_answers, current_query, label in zip(
        examples["history_query"],
        examples["history_answer"],
        examples["query"],
        examples["rewrite_prompt"]
    ):
        input_id = build_input_topiocqa(
            history_queries=history_queries,
            history_answers=history_answers,
            current_query=current_query,
            tokenizer=tokenizer,
            max_concat_length=max_concat_length,
            max_query_length=max_query_length,
            per_turn_max_length=per_turn_max_length
        )

        input_ids.append(input_id)
        attention_mask.append([1] * len(input_id))
        targets.append(label)

    labels = tokenizer(
        text_target=targets,
        truncation=True,
        max_length=target_max_length,
        padding=False,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels["input_ids"]
    }


def build_input(context, tokenizer, max_concat_length=512, max_query_length=64):
    
    # 1️⃣ 编码 rewrite
    rewrite_ids = tokenizer.encode(
        "REWRITE: " ,  #+ rewrite
        add_special_tokens=True,
        max_length=max_query_length,
        truncation=True
    )

    # 2️⃣ 倒序拼 context
    history_ids = []

    for utt in reversed(context):
        utt_ids = tokenizer.encode(
            "CTX: " + utt + " <sep>",
            add_special_tokens=False,
            truncation=True,
            max_length=128  # 单条限制（防爆）
        )

        if len(history_ids) + len(utt_ids) > max_concat_length:
            remain = max_concat_length - len(history_ids)
            if remain > 0:
                history_ids = utt_ids[-remain:] + history_ids
            break
        else:
            history_ids = utt_ids + history_ids

    # 3️⃣ 拼接
    input_ids = history_ids + rewrite_ids

    return input_ids


# 加载数据集
dataset = load_dataset("json", data_files={"train": "/home/zhengjiaying/project/RAG-test/topicoqa/papers_full_train_re.jsonl",
                                                 "validation": "/home/zhengjiaying/project/RAG-test/topicoqa/devdata/select_papers.jsonl"},
                       cache_dir="./cache",  # 缓存数据集，避免重复解析JSON
                       num_proc=8)


# 初始化tokenizer（flan-t5-base）
# 加载 FLAN-T5 模型和 Tokenizer
model_name = "/home/zhengjiaying/project/RAG-test/Flan-T5-large"

# model = T5RewardModel(model_name)
model =  AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(tokenizer.truncation_side)

# tokenizer.truncation_side = "left"
special_tokens_dict = {"sep_token": "<sep>"}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))


model.train()

# # 应用预处理
# def filter_has_retrieval(example):
#     return len(example.get("pos_docs_pids", [])) > 0

# dataset = dataset.filter(
#     filter_has_retrieval,
#     num_proc=8
# )

processed_dataset = dataset.map(
    preprocess_function_topiocqa,
    batched=True,
    batch_size=1000,  # 批量预处理，减少进程切换
    remove_columns=dataset["train"].column_names,
    num_proc=4,  # 16个进程预处理（核心数够就设32）
    load_from_cache_file=True,  # 缓存预处理结果
    fn_kwargs={
        "tokenizer": tokenizer,
        "max_concat_length": 512,
        "max_query_length": 64,
        "per_turn_max_length": 128
    }
)

  
    
class DataCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        input_features = []
        label_features = []

        for f in features:
            input_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"]
            })
            
            label_features.append(f["labels"])

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        max_label_len = max(len(x) for x in label_features)
        padded_labels = []
        for x in label_features:
            padded = x + [self.label_pad_token_id] * (max_label_len - len(x))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

        # batch = {
        #     "input_ids": batch["input_ids"],
        #     "attention_mask": batch["attention_mask"],           
        #     "labels": torch.tensor(labels, dtype=torch.float32),
        # }

        # return batch
# ------------------- 3. 自定义评估指标（偏好准确率） -------------------    
    
def compute_metrics3(eval_pred):
    """
    Reward model evaluation metrics
    兼容 Trainer 的各种 prediction 结构
    logits 预期 shape:
        (N, 2) 或 (num_batches, batch, 2)
    """

    # 1️⃣ 拿 predictions
    preds = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]

    # tuple 取第一个
    if isinstance(preds, tuple):
        preds = preds[0]

    # tensor → numpy
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.array(preds)

    # 2️⃣ shape 统一
    # (num_batches, batch, 2) → (N, 2)
    if preds.ndim == 3:
        preds = preds.reshape(-1, preds.shape[-1])

    # 防止异常
    if preds.shape[-1] != 2:
        return {
            "eval_preference_accuracy": 0.0,
            "eval_avg_chosen_score": 0.0,
            "eval_avg_rejected_score": 0.0,
        }

    # 3️⃣ 拆 chosen / rejected
    chosen = preds[:, 0]
    rejected = preds[:, 1]

    # 4️⃣ NaN / inf 处理
    chosen = np.nan_to_num(chosen, nan=0.0, posinf=1.0, neginf=-1.0)
    rejected = np.nan_to_num(rejected, nan=0.0, posinf=1.0, neginf=-1.0)

    # 5️⃣ 计算指标
    preference_accuracy = float(np.mean(chosen > rejected))
    avg_chosen_score = float(np.mean(chosen))
    avg_rejected_score = float(np.mean(rejected))

    return {
        "eval_preference_accuracy": preference_accuracy,
        "eval_avg_chosen_score": avg_chosen_score,
        "eval_avg_rejected_score": avg_rejected_score,
    }

model.config.pad_token_id = tokenizer.pad_token_id

# 训练参数（原生Trainer风格，支持save_strategy="best"）
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/zhengjiaying/project/RAG-test/t5-checkpoint/sft_flant5_large_redata_new_topiocqa_gpt_0408_3",
    do_train = True,
    do_eval = False,
    eval_strategy='no',
    per_device_train_batch_size=4,
    num_train_epochs=8,
    learning_rate=5e-4,# 不懂了。。。。
    max_grad_norm=1.0,
    save_total_limit=10,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine', # 'linear'
    bf16=True,
    gradient_checkpointing=False,
    logging_steps=710,
    save_strategy="steps",         # 改为按步数保存
    save_steps=710,                # 每500步保存一次（和eval_steps一致）
    remove_unused_columns=False,  # 关键：保留labels字段用于分组
    gradient_accumulation_steps=8,  # 梯度累积，有效batch=4*4=16，GPU计算量拉满
    dataloader_num_workers=16,  # 16个工作进程加载数据（CPU核心数的1/2）
    dataloader_pin_memory=True,  # 内存锁定，加速CPU→GPU传输
    dataloader_prefetch_factor=2,
    report_to='none',
)


# data_collator = PairwiseRewardDataCollator(tokenizer,512)
# data_collator = DataCollator(tokenizer,576)

# ------------------- 5. 初始化Trainer -------------------
trainer = Seq2SeqTrainer(
    model=model,
    # processing_class=tokenizer,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, pad_to_multiple_of=8, label_pad_token_id=-100, max_length=576
        ),
)

# ------------------- 6. 开始训练（和RewardTrainer效果一致，无decoder报错） -------------------
trainer.train()
