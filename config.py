#!/usr/bin/env python
# author = 'ZZH'
# time = 2023/6/23
# project = config
import argparse
import torch


def get_args():
    model_dict = {
        "codet5p-220m": 'Salesforce/codet5p-220m',
        "codet5p-770m": "Salesforce/codet5p-770m",
        "codet5p-2b": "Salesforce/codet5p-2b",
        "codet5p-6b": "Salesforce/codet5p-6b",
        "codet5p-16b": "Salesforce/codet5p-16b",
        "llama7b":"decapoda-research/llama-7b-hf"
    }

    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--max_source_len', default=150, type=int)
    parser.add_argument('--max_target_len', default=150, type=int)
    # parser.add_argument('--cache-data', default='cache_data/summarize_java', type=str)
    parser.add_argument('--load', default='llama7b', type=str)
    parser.add_argument('--beam_size', default=1, type=int)

    # Training
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--warmup_steps', default=100, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--grad_acc_steps', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--early_stop', default=5, type=int)

    # Logging and stuff
    parser.add_argument('--data_name', default="text2code", type=str, help='data.json dataset')
    parser.add_argument('--checkpoint_dir', default=None, type=str, help="path/to/lora_dir")
    parser.add_argument('--save_dir', default="temp", type=str)
    parser.add_argument('--type', default="lora", choices=['lora', 'full'], type=str, help="lora or full")
    parser.add_argument('--lora_r', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--target_modules', default=["q", "v", "o"],
                        help=" The amount of parameters is greater than or equal to codet5-2b['q_proj','k_proj' 'v_proj', 'o_proj']")
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_path = model_dict[args.load]
    args.codet5_b_flag = True if args.load in ["codet5p-2b", "codet5p-6b", "codet5p-16b"] else False
    if args.type == 'full':
        del args.target_modules
        del args.lora_r
        del args.lora_alpha
    if args.type == 'lora' and (args.codet5_b_flag or "llama" in args.load):
        target_modules = args.target_modules
        target_modules = [module + '_proj' for module in target_modules]
        args.target_modules = target_modules
    return args
