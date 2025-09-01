# train_lora.py  -- corrected dataset-loading version
# Run in Colab after installing: transformers, datasets, accelerate, peft, bitsandbytes (optional)

import os
import random
import inspect
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
# BitsAndBytesConfig may not exist in all versions; import guarded
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    HAS_BNB = False

# PEFT imports
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType,
)

# -----------------------
# USER CONFIG (edit here)
# -----------------------
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/PathoBuddy/data/lora_dataset.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/PathoBuddy/lora_weights"
BASE_MODEL = "EleutherAI/pythia-1b-deduped"   # good tradeoff for Colab free
SEED = 42
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_LENGTH = 512
VAL_SPLIT = 0.10
LR = 2e-4
SAVE_TOTAL_LIMIT = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# 1) Load & split dataset (CORRECTED)
# -----------------------
print("Loading dataset from:", DATA_PATH)
# <-- CORRECTION: do NOT use a dict key named "all". Load single-file split="train" then split it.
ds = load_dataset("json", data_files=DATA_PATH, split="train")
ds = ds.shuffle(seed=SEED)
if VAL_SPLIT > 0:
    split = ds.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_raw = split["train"]
    val_raw = split["test"]
else:
    train_raw = ds
    val_raw = None

print(f"Train size: {len(train_raw)}", f"Val size: {len(val_raw) if val_raw is not None else 0}")

# -----------------------
# 2) Tokenizer & special tokens
# -----------------------
print("Loading tokenizer for:", BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)

specials = {}
if tokenizer.pad_token is None:
    specials["pad_token"] = "<|pad|>"
if tokenizer.eos_token is None:
    specials["eos_token"] = "<|endoftext|>"

if specials:
    tokenizer.add_special_tokens(specials)
    print("Added special tokens:", specials)

# -----------------------
# 3) Preprocess function (mask prompt in labels)
# -----------------------
def build_prompt(prompt_text: str) -> str:
    # Keep prompt format consistent with inference
    return f"Question: {prompt_text}\nAnswer:"

def preprocess_batch(examples):
    # batched mapping
    prompts = examples.get("prompt") or examples.get("question") or []
    completions = examples.get("completion") or examples.get("response") or []
    eos = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"

    input_ids_batch = []
    attention_batch = []
    labels_batch = []

    for pr, comp in zip(prompts, completions):
        prompt_fmt = build_prompt(pr)
        full = f"{prompt_fmt} {comp} {eos}"
        tok = tokenizer(full, truncation=True, padding="max_length", max_length=MAX_LENGTH)

        # mask prompt tokens
        prompt_ids = tokenizer(prompt_fmt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        labels = tok["input_ids"].copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        # mask padding positions
        labels = [lab if am==1 else -100 for lab, am in zip(labels, tok["attention_mask"])]

        input_ids_batch.append(tok["input_ids"])
        attention_batch.append(tok["attention_mask"])
        labels_batch.append(labels)

    return {"input_ids": input_ids_batch, "attention_mask": attention_batch, "labels": labels_batch}

print("Tokenizing (batched)...")
train_tok = train_raw.map(preprocess_batch, batched=True, batch_size=32, remove_columns=train_raw.column_names)
val_tok = None
if val_raw is not None:
    val_tok = val_raw.map(preprocess_batch, batched=True, batch_size=32, remove_columns=val_raw.column_names)

print("Sample -100 labels in first example:", sum(1 for x in train_tok[0]["labels"] if x == -100))

# -----------------------
# 4) Quantization config (optional)
# -----------------------
bnb_config = None
if HAS_BNB:
    print("bitsandbytes available -> using 4-bit quantization config")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
else:
    print("bitsandbytes not available - loading full precision (may OOM).")

# -----------------------
# 5) Load base model
# -----------------------
print("Loading base model (this may take a while)...")
model_kwargs = dict(
    torch_dtype=torch.float16,
    trust_remote_code=False,
)
if bnb_config is not None:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", **model_kwargs
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", **model_kwargs
    )

# If tokenizer changed (added specials) resize embeddings
if len(tokenizer) != model.get_input_embeddings().weight.size(0):
    model.resize_token_embeddings(len(tokenizer))

# Prepare for kbit training (QLoRA style)
try:
    model = prepare_model_for_kbit_training(model)
except Exception:
    # If prepare_model_for_kbit_training not compatible, continue (PEFT may still work)
    print("Warning: prepare_model_for_kbit_training failed or unavailable; continuing without it.")

# -----------------------
# 6) LoRA config & wrap
# -----------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------
# 7) TrainingArguments (compat-safe)
# -----------------------
# Some older transformers versions choke on certain kwargs (e.g. evaluation_strategy).
# We'll detect if evaluation_strategy is supported and include it only if safe.

ta_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=SAVE_TOTAL_LIMIT,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

# safe injection of evaluation_strategy if supported
sig = inspect.signature(TrainingArguments.__init__)
if "evaluation_strategy" in sig.parameters:
    ta_kwargs["evaluation_strategy"] = "epoch"

training_args = TrainingArguments(**ta_kwargs)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok if ("evaluation_strategy" in ta_kwargs) else None,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

model.config.use_cache = False

# -----------------------
# 8) Train
# -----------------------
print("Starting training...")
trainer.train()

# -----------------------
# 9) If evaluation_strategy wasn't used, run manual eval now
# -----------------------
if "evaluation_strategy" not in ta_kwargs and val_tok is not None:
    try:
        print("\nManual evaluation on validation set:")
        res = trainer.evaluate(eval_dataset=val_tok)
        print(res)
    except Exception as e:
        print("Manual evaluation failed:", e)

# -----------------------
# 10) Save LoRA + tokenizer
# -----------------------
print("Saving LoRA adapter and tokenizer to:", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -----------------------
# 11) Quick generation sanity-check
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)

def generate_prompt(prompt_text, max_new_tokens=120, concise=False):
    prompt = f"Question: {prompt_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if concise:
        gen_kwargs.update({"do_sample": False, "num_beams": 1})
    else:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "num_beams": 1,
        })

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    raw = tokenizer.decode(out[0], skip_special_tokens=False)
    eos = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if eos and eos in raw:
        raw = raw.split(eos)[0]
    return raw.strip()

print("\n=== Quick sanity check on a few validation items ===")
if val_raw is not None and len(val_raw) > 0:
    sample_idxs = random.sample(range(len(val_raw)), min(5, len(val_raw)))
    for i in sample_idxs:
        q = val_raw[i]["prompt"]
        print("PROMPT:", q)
        print("-> concise:", generate_prompt(q, max_new_tokens=40, concise=True))
        print("-> descriptive:", generate_prompt(q, max_new_tokens=180, concise=False))
        print("-"*80)
else:
    print("No validation set to test generation.")

print("\nDone. LoRA saved at:", OUTPUT_DIR)
