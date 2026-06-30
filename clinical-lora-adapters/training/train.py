"""LoRA / QLoRA fine-tuning for one clinical adapter.

What this produces: a small adapter folder (adapters/<name>/) containing only the LoRA
weights — a few tens of MB — NOT a full 7B checkpoint. That is the entire economic point:
N specialties cost one base model + N tiny adapters, instead of N full models.

Key choices:
- QLoRA: base weights frozen and quantized to 4-bit (when CUDA + bitsandbytes available);
  only the low-rank adapter is trained. Fits a 7B model on a single 16 GB GPU.
- Completion-only loss: we mask the prompt tokens (labels = -100) so the model is graded
  only on the target it should generate, not on reproducing the instruction.
- Format comes from common.prompts so train-time and serve-time wrapping are identical.

Usage:
    python training/train.py --adapter cardiology-summary
    python training/train.py --adapter radiology-extract --epochs 3 --lr 2e-4
    BASE_MODEL=sshleifer/tiny-gpt2 USE_4BIT=0 python training/train.py --adapter cardiology-summary   # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import prompts  # noqa: E402
from common.config import (  # noqa: E402
    ADAPTERS,
    BASE_MODEL,
    DEFAULT_LORA,
    DEFAULT_TRAIN,
    USE_4BIT,
    get_adapter,
)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        # No dedicated pad token on most decoder LMs; reuse eos. We right-pad and mask
        # pads in the attention mask, so this is safe for causal LM training.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(model_name: str):
    """Load the frozen base. 4-bit QLoRA on CUDA; plain fp16/fp32 elsewhere."""
    from transformers import AutoModelForCausalLM

    use_4bit = USE_4BIT and torch.cuda.is_available()
    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.cuda()
    model.config.use_cache = False  # required with gradient checkpointing / training
    return model


def attach_lora(model, hp):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=hp.r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        bias=hp.bias,
        task_type="CAUSAL_LM",
        target_modules=hp.target_modules,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()  # prints the ~0.1-1% trainable headline number
    return model


def build_dataset(tokenizer, spec, max_seq_len: int):
    """Tokenize each record into input_ids + labels with the prompt masked out."""
    records = json.loads(spec.data_path.read_text())
    examples = []
    for rec in records:
        input_text = rec[spec.input_key]
        target = rec[spec.target_key]

        full = prompts.format_training_text(tokenizer, spec, input_text, target)
        prompt_only = prompts.format_inference_prompt(tokenizer, spec, input_text)

        full_ids = tokenizer(full, truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_only, truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]

        labels = list(full_ids)
        mask_len = min(len(prompt_ids), len(full_ids))
        for i in range(mask_len):
            labels[i] = -100  # don't compute loss on the instruction/prompt
        examples.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    print(f"Built {len(examples)} training examples (prompt-masked completion loss).")
    return examples


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)  # pads never contribute to loss
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def train(adapter_name: str, overrides: dict) -> Path:
    from transformers import Trainer, TrainingArguments

    spec = get_adapter(adapter_name)
    hp = DEFAULT_TRAIN
    for k, v in overrides.items():
        if v is not None:
            setattr(hp, k, v)

    print(f"== Training adapter '{spec.name}' ({spec.task_type}) on base {BASE_MODEL} ==")
    tokenizer = load_tokenizer(BASE_MODEL)
    model = attach_lora(load_base_model(BASE_MODEL), DEFAULT_LORA)
    dataset = build_dataset(tokenizer, spec, hp.max_seq_len)

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(spec.output_dir / "_trainer"),
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=hp.grad_accum_steps,
        learning_rate=hp.learning_rate,
        warmup_ratio=hp.warmup_ratio,
        logging_steps=hp.logging_steps,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),
        report_to="none",
        seed=hp.seed,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    trainer.train()

    # Save ONLY the adapter weights + the tokenizer + a manifest tying it to its task.
    model.save_pretrained(str(spec.output_dir))
    tokenizer.save_pretrained(str(spec.output_dir))
    (spec.output_dir / "adapter_meta.json").write_text(
        json.dumps(
            {"name": spec.name, "base_model": BASE_MODEL, "task_type": spec.task_type,
             "specialty": spec.specialty, "n_examples": len(dataset)},
            indent=2,
        )
    )
    print(f"Saved adapter to {spec.output_dir}")
    return spec.output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, choices=list(ADAPTERS))
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None, dest="learning_rate")
    ap.add_argument("--batch-size", type=int, default=None, dest="per_device_batch_size")
    ap.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")
    args = vars(ap.parse_args())
    adapter = args.pop("adapter")
    train(adapter, args)


if __name__ == "__main__":
    main()
