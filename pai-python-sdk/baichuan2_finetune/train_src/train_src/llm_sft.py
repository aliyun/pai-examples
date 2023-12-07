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
import inspect
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from functools import partial

import json
import numpy as np
import torch
from transformers import BitsAndBytesConfig, GenerationConfig
from .utils import (SftArguments, dataset_map, get_dataset, get_model_tokenizer,
                   get_preprocess)

from swift import (LoraConfig, LoRAConfig, Seq2SeqTrainer,
                   Seq2SeqTrainingArguments, Swift, get_logger)
from swift.utils import (add_version_to_work_dir, broadcast_string,
                         check_json_format, compute_nlg_metrics,
                         data_collate_fn, find_all_linear_for_lora,
                         get_dist_setting, is_ddp_plus_mp, is_dist, is_master,
                         parse_args, plot_images, print_example,
                         print_model_info, seed_everything, show_layers,
                         sort_by_max_length, stat_dataset)

logger = get_logger()


def llm_sft(args: SftArguments) -> None:
    logger.info(f'args: {args}')
    print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, '
          f'world_size: {world_size}, local_world_size: {local_world_size}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True}
    if is_dist() and not is_ddp_plus_mp():
        kwargs['device_map'] = {'': local_rank}
    else:
        kwargs['device_map'] = 'auto'
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        kwargs['quantization_config'] = quantization_config
    if args.model_type.startswith('qwen'):
        kwargs['use_flash_attn'] = args.use_flash_attn
    

    kwargs["model_dir"] = os.environ.get("PAI_INPUT_PRETRAINED_MODEL")
    args.model_name_or_path = os.environ.get("PAI_INPUT_PRETRAINED_MODEL")
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    # ### Preparing LoRA
    if args.resume_from_ckpt is None:
        if args.sft_type == 'lora':
            if 'ALL' in args.lora_target_modules:
                assert len(args.lora_target_modules) == 1
                args.lora_target_modules = find_all_linear_for_lora(
                    model, args.quantization_bit, args.model_type)
                logger.info(
                    f'Setting lora_target_modules: {args.lora_target_modules}')
            lora_kwargs = {}
            if args.tuner_bankend == 'peft':
                global LoRAConfig
                LoRAConfig = LoraConfig
                lora_kwargs['task_type'] = 'CAUSAL_LM'
            lora_config = LoRAConfig(
                r=args.lora_rank,
                target_modules=args.lora_target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout_p,
                **lora_kwargs)
            model = Swift.prepare_model(model, lora_config)
            logger.info(f'lora_config: {lora_config}')
    else:
        model = Swift.from_pretrained(
            model, args.resume_from_ckpt, is_trainable=True)

    show_layers(model)
    print_model_info(model)
    logger.info(model)

    # ### Loading Dataset
    train_dataset, val_dataset = get_dataset(
        args.dataset.split(','), args.dataset_test_ratio,
        args.dataset_split_seed)
    if args.train_dataset_sample >= 0:
        val_dataset_sample = max(
            int(args.train_dataset_sample * args.dataset_test_ratio), 1)
        train_idxs = np.random.permutation(args.train_dataset_sample)
        train_dataset = train_dataset.select(train_idxs)
        if val_dataset.shape[0] > val_dataset_sample:
            val_idxs = np.random.permutation(val_dataset_sample)
            val_dataset = val_dataset.select(val_idxs)
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    preprocess_func = get_preprocess(args.template_type, tokenizer,
                                     args.system, args.max_length)
    train_dataset = dataset_map(train_dataset, preprocess_func)
    val_preprocess_func = preprocess_func
    if args.predict_with_generate:
        val_preprocess_func = partial(preprocess_func, generation_mode=True)
    val_dataset = dataset_map(val_dataset, val_preprocess_func)
    if args.test_oom_error:
        train_dataset = sort_by_max_length(train_dataset, 20000)
    # Data analysis
    stat_dataset(train_dataset)
    stat_dataset(val_dataset)
    data_collator = partial(data_collate_fn, tokenizer=tokenizer)
    print_example(train_dataset[0], tokenizer)

    # ### Setting training_args
    output_dir = args.output_dir
    # if is_master():
    #     output_dir = add_version_to_work_dir(args.output_dir)
    # if is_dist():
    #     # Make sure to set the same output_dir when using DDP.
    #     output_dir = broadcast_string(output_dir)
    # check ms-swift version
    parameters = inspect.signature(
        Seq2SeqTrainingArguments.__init__).parameters
    for k in ['only_save_model', 'train_sampler_random']:
        if k not in parameters:
            raise ValueError(
                f'The `{k}` parameter is invalid. '
                'You can resolve this warning by upgrading ms-swift or installing from source.'
            )
    # training_args
    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        max_length=None,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty)
    logger.info(f'generation_config: {generation_config}')
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.environ.get("PAI_OUTPUT_TENSORBOARD"),
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model='rouge-l'
        if args.predict_with_generate else 'loss',
        greater_is_better=args.predict_with_generate,
        sortish_sampler=True,
        optim=args.optim,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        push_hub_strategy=args.push_hub_strategy,
        hub_token=args.hub_token,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_ckpt,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        predict_with_generate=args.predict_with_generate,
        generation_config=generation_config,
        local_rank=local_rank,
        only_save_model=args.only_save_model,
        train_sampler_random=args.train_sampler_random)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if is_dist():
        # Compatible with https://github.com/huggingface/transformers/pull/25903
        training_args._frozen = False
        if args.gradient_checkpointing:
            training_args.ddp_find_unused_parameters = False
            training_args.ddp_broadcast_buffers = False
        else:
            training_args.ddp_find_unused_parameters = True
            training_args.ddp_broadcast_buffers = True

    logger.info(f'training_args: {training_args}')

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_nlg_metrics, tokenizer=tokenizer)
        if args.predict_with_generate else None,
    )
    if is_master():
        for args_obj, fname in zip([args, training_args],
                                   ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'w') as f:
                json.dump(
                    check_json_format(args_obj.__dict__),
                    f,
                    ensure_ascii=False,
                    indent=2)
    trainer.train(training_args.resume_from_checkpoint)
    logger.info(
        f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
    # /ml/output/model/{model_type}/{version} => 挂载到 EAS
    trainer.save_model()

    # ### Visualization
    if is_master():
        # images_dir = os.path.join(output_dir, 'images')
        # logger.info(f'images_dir: {images_dir}')
        # tb_dir = os.path.join(output_dir, 'runs')
        # folder_name = os.listdir(tb_dir)[0]
        # tb_dir = os.path.join(tb_dir, folder_name)
        # plot_images(images_dir, tb_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer._add_patterns_to_gitignores(['images/'])
            trainer.push_to_hub()


if __name__ == '__main__':
    args, remaining_argv = parse_args(SftArguments)
    args.init_argument()
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_sft(args)
