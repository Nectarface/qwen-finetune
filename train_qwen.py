#!/usr/bin/env python3
"""
Full Fine-Tuning Script for Qwen3.5-9B-Base
Uses DeepSpeed ZeRO-3 for distributed training across multiple GPUs
"""

import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to training JSONL')
    parser.add_argument('--output', type=str, default='/workspace/output/qwen3.5-9b-finetuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    args = parser.parse_args()

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset
    import torch

    print("="*60)
    print("QWEN 3.5 9B FULL FINE-TUNING")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} gradient accumulation")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")
    print("="*60)

    # Load model and tokenizer
    print("\nLoading Qwen3.5-9B-Base...")
    model_name = "Qwen/Qwen3.5-9B-Base"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    print(f"Model parameters: {model.num_parameters():,}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    ds = load_dataset('json', data_files=args.dataset, split='train')
    print(f"Dataset size: {len(ds)} examples")

    # Tokenize
    def tokenize_conversation(example):
        """Convert messages format to single text for training"""
        messages = example.get('messages', [])

        # Build conversation text
        text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                text_parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                text_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                text_parts.append(f"<|assistant|>\n{content}")

        text = "\n".join(text_parts)

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    print("Tokenizing dataset...")
    ds = ds.map(
        tokenize_conversation,
        remove_columns=ds.column_names,
        num_proc=8,
        desc="Tokenizing"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    os.makedirs(args.output, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed="ds_config.json",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
