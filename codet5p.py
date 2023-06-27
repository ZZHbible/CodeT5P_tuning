#!/usr/bin/env python
# author = 'ZZH'
# time = 2023/6/19
# project = codet5p
# !/usr/bin/env python
# author = 'ZZH'
# time = 2023/6/15
# project = codet5

"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF data format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
import pprint
import numpy
import pandas as pd
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, AutoConfig
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import logging
import math
from tqdm import tqdm
from utils import get_bleu_socre, load_model, load_tokenize_data
from config import get_args

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(args, model, train_data, eval_data=None):
    print(f"Starting main loop")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.epochs),
    )

    # Start training
    train_example_num = len(train_data)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_example_num)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
    logger.info("  Num epoch = %d", args.epochs)
    global_step, best_bleu, best_loss = 0, -1, 1e6
    count = 0

    for cur_epoch in range(int(args.epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        model.train()
        for step, batch in enumerate(bar):
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)

            total_loss = outputs.loss
            total_loss.backward()

            tr_loss += total_loss.item()
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
            wandb.log({'train_loss': train_loss.item()})
            bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

        if eval_data:
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            logger.info("***** Running evaluation  *****")
            logger.info("  Num examples = %d", len(eval_data))
            logger.info("  Batch size = %d", args.eval_batch_size)
            logger.info("  Num epoch = %d", cur_epoch)
            model.eval()
            eval_loss, batch_num = 0, 0
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch

                with torch.no_grad():
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss
                eval_loss = eval_loss + loss.item()
                batch_num += 1
            model.train()
            eval_loss = eval_loss / batch_num
            result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                      'global_step': global_step + 1,
                      'eval_loss': round(eval_loss, 5)}
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  " + "*" * 20)

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            model.eval()
            all_outputs = []
            # Batching
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Generating outputs", ):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                generate_kwargs = {
                    "input_ids": source_ids,
                    "attention_mask": source_mask,
                    "num_beams": args.beam_size,
                    "max_new_tokens": args.max_target_len,
                    # "eos_token_id" :args.tokenizer.eos_token_id,
                }
                outputs = model.generate(**generate_kwargs)
                all_outputs.extend(outputs.cpu().numpy())
            hyp_list = [
                args.tokenizer.decode(
                    output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for output_id in all_outputs
            ]

            assert len(args.eval_data) == len(hyp_list)
            df = pd.DataFrame(hyp_list)
            df.to_csv("hyp_temp.csv", index=False, header=None)
            df = pd.DataFrame(args.eval_data)
            df.to_csv("ref_temp.csv", index=False, header=None)

            bleu4, acc = get_bleu_socre("ref_temp.csv", "hyp_temp.csv")

            if bleu4 >= best_bleu:
                df = pd.DataFrame(hyp_list)
                df.to_csv(os.path.join(args.save_dir, "preds.csv"), index=False, header=None)
                df = pd.DataFrame(args.eval_data)
                df.to_csv(os.path.join(args.save_dir, "golds.csv"), index=False, header=None)
                count = 0
                logger.info("  Best bleu:%s", bleu4)
                logger.info("  " + "*" * 20)
                best_bleu = bleu4
                # Save best checkpoint for best bleu
                output_dir_bleu = os.path.join(args.save_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir_bleu):
                    os.makedirs(output_dir_bleu)

                model.save_pretrained(output_dir_bleu)
            else:
                count += 1
                if count == args.early_stop:
                    break
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()


def test(args, test_data):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if args.codet5_b_flag:
        config.pad_token_id = args.tokenizer.pad_token_id
        # config.decoder_start_token_id = tokenizer.pad_token_id
        config.decoder_start_token_id = args.tokenizer.bos_token_id
    if args.type == 'lora':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, config=config, trust_remote_code=True,
                                                      torch_dtype=torch.float16) if args.fp16 else AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path, config=config, trust_remote_code=True)
        model = PeftModel.from_pretrained(
            model,
            os.path.join(args.save_dir, 'checkpoint-best-bleu'),
            is_trainable=True
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, 'checkpoint-best-bleu'),
                                                      config=config, trust_remote_code=True,
                                                      torch_dtype=torch.float16) if args.fp16 else AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path, config=config, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8)

    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(test_data))
    logger.info("  Batch size = %d", 8)
    model.eval()
    all_outputs = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch
        generate_kwargs = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "num_beams": args.beam_size,
            "max_new_tokens": args.max_target_len,
            # "eos_token_id" :args.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
            all_outputs.extend(outputs.cpu().numpy())
    hyp_list = [
        args.tokenizer.decode(
            output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for output_id in all_outputs
    ]
    with open('output.txt', 'w') as f:
        for hyp in hyp_list:
            f.write(hyp + '\n')


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    model = load_model(args)
    train_data, eval_data, test_data = load_tokenize_data(args)
    run_training(args, model, train_data, eval_data)
    del model
    # test
    test(args, test_data)


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)
    wandb.init("CodeT5P_training", name=args.type + args.load + args.data_name)
    main(args)
