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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import BitsAndBytesConfig, GenerationConfig, TextStreamer
from utils import (InferArguments, get_dataset, get_model_tokenizer,
                   get_preprocess)

from swift import Swift, get_logger
from swift.utils import (inference, parse_args, print_model_info,
                         seed_everything, show_layers)

logger = get_logger()


def merge_lora(args: InferArguments) -> None:
    assert args.sft_type == 'lora'
    # ### Loading Model and Tokenizer
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, device_map='cpu')

    # ### Preparing LoRA
    model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
    Swift.merge_and_unload(model)

    ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
    merged_lora_path = os.path.abspath(
        os.path.join(ckpt_dir, f'{ckpt_name}-merged'))
    logger.info(f'merged_lora_path: `{merged_lora_path}`')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.sft_type = 'full'
    args.ckpt_dir = merged_lora_path
    if not os.path.exists(args.ckpt_dir):
        logger.info('Saving merged weights...')
        model.model.save_pretrained(args.ckpt_dir)
        tokenizer.save_pretrained(args.ckpt_dir)
        logger.info('Successfully merged LoRA.')
    else:
        logger.info('The weight directory for the merged LoRa already exists, '
                    'skipping the saving process.')


def llm_infer(args: InferArguments) -> None:
    if args.merge_lora_and_save:
        merge_lora(args)
    logger.info(f'args: {args}')
    if not os.path.isdir(args.ckpt_dir):
        raise ValueError(f'Please enter a valid ckpt_dir: {args.ckpt_dir}')
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
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

    if args.sft_type == 'full':
        kwargs['model_dir'] = args.ckpt_dir
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    # ### Preparing LoRA
    if args.sft_type == 'lora':
        model = Swift.from_pretrained(
            model, args.ckpt_dir, inference_mode=True)

    show_layers(model)
    print_model_info(model)

    # ### Inference
    preprocess_func = get_preprocess(args.template_type, tokenizer,
                                     args.system, args.max_length)
    streamer = None
    if args.use_streamer:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    if args.save_generation_config:
        generation_config.save_pretrained(args.ckpt_dir)
    model.generation_config = generation_config

    if args.eval_human:
        while True:
            query = input('<<< ')
            data = {'query': query}
            input_ids = preprocess_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer)
    else:
        _, val_dataset = get_dataset(
            args.dataset.split(','), args.dataset_test_ratio,
            args.dataset_split_seed)
        mini_val_dataset = val_dataset.select(
            range(min(args.show_dataset_sample, val_dataset.shape[0])))
        for data in mini_val_dataset:
            response = data['response']
            data['response'] = None
            input_ids = preprocess_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer)
            print()
            print(f'[LABELS]{response}')
            print('-' * 80)
            # input('next[ENTER]')


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments)
    args.init_argument()
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_infer(args)
