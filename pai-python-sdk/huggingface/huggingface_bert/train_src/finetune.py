#  Copyright 2023 Alibaba, Inc. or its affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    HfArgumentParser,
)
import numpy as np
import evaluate


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def train():
    # 通过环境变量获取预训练模型地址, 训练数据，以及模型保存地址
    model_name_or_path = os.environ.get("PAI_INPUT_MODEL", "bert-base-cased")
    input_train_data = os.environ.get("PAI_INPUT_TRAIN_DATA")
    output_dir = os.environ.get("PAI_OUTPUT_MODEL", "./output")

    # 使用环境变量获取训练作业超参
    num_train_epochs = int(os.environ.get("PAI_HPS_EPOCHS", 2))
    save_strategy = os.environ.get("PAI_HPS_SAVE_STRATEGY", "epoch")

    print("Loading Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=5
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print("Loading dataset from disk...")
    dataset = load_from_disk(input_train_data)
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        ),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    training_args = TrainingArguments(
        output_dir=output_dir,
        # 使用环境变量获取训练作业超参
        num_train_epochs=num_train_epochs,
        # 使用环境变量获取训练作业保存策略
        save_strategy=save_strategy,
    )
    print("TrainingArguments: {}".format(training_args.to_json_string()))
    metric = evaluate.load("accuracy")

    print("Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Saving Model...")
    trainer.save_model()


if __name__ == "__main__":
    train()
