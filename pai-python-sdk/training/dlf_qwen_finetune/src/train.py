import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import oss2
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


@dataclass
class ScriptArgs:
    model_name_or_path: str
    output_dir: str
    logging_dir: str
    checkpoint_dir: str

    learning_rate: float
    batch_size: int
    seq_length: int
    num_train_epochs: int
    system_prompt: str


def load_oss_dataset() -> Dataset:
    bucket = oss2.Bucket(
        auth=oss2.AnonymousAuth(),
        endpoint="oss-cn-hangzhou.aliyuncs.com",
        bucket_name="pai-quickstart-cn-hangzhou",
    )
    key = "public_datasets/Chinese-medical-dialogue-data/chinese_medical_train_sampled.json"

    train_path = "/tmp/sampled_train.json"
    bucket.get_object_to_file(key=key, filename=train_path)
    return load_dataset("json", data_files=train_path, split="train")


def load_paimon_dataset(input_uri: str) -> Dataset:
    pass


def formatting_prompts(system_prompt: str, tokenizer) -> Callable[[List[Dict]], List]:
    system_prompt = system_prompt or "You are a helpful assistant"

    def _format_func(examples: List[Dict]) -> List:
        output_text = []
        for i in range(len(examples["instruction"])):
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": examples["instruction"][i],
                },
                {
                    "role": "assistant",
                    "content": examples["output"][i],
                },
            ]
            output_text.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return output_text

    return _format_func


def load_model_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8

    return model, tokenizer


def init_script_args() -> ScriptArgs:
    # loading hyperparameters
    hps = json.loads(os.getenv("PAI_HPS")) if os.getenv("PAI_HPS") else {}
    learning_rate = float(hps.get("learning_rate", 1.41e-5))
    batch_size = int(hps.get("batch_size", 2))
    seq_length = int(hps.get("seq_length", 512))
    num_train_epochs = int(hps.get("num_train_epochs", 1))
    logging_dir = hps.get("logging_dir", "/ml/output/tensorboard/")
    system_prompt = hps.get("system_prompt", "You are a helpful assistant.")
    checkpoints_dir = "/ml/output/checkpoints/"
    logging_dir = "/ml/output/tensorboard/"

    return ScriptArgs(
        # model_name_or_path="qwen2_model",
        model_name_or_path="/ml/input/data/model/",
        output_dir="/ml/output/model",
        logging_dir=logging_dir,
        checkpoint_dir=checkpoints_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seq_length=seq_length,
        num_train_epochs=num_train_epochs,
        system_prompt=system_prompt,
    )


def train():
    script_args = init_script_args()
    dataset = load_oss_dataset()
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=script_args.model_name_or_path,
    )
    bf16_flag = torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir=script_args.checkpoint_dir,
        per_device_train_batch_size=script_args.batch_size,
        learning_rate=script_args.learning_rate,
        num_train_epochs=script_args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        logging_steps=10,
        report_to="tensorboard",
        logging_dir=script_args.logging_dir,
        bf16=bf16_flag,
        fp16=not bf16_flag,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=dataset,
        formatting_func=formatting_prompts(
            script_args.system_prompt, tokenizer=tokenizer
        ),
    )
    trainer.train()

    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)


if __name__ == "__main__":
    train()
