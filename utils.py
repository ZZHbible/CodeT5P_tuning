# -*- coding: utf-8 -*-
import glob
import json
import os
import torch
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from bleu import compute_bleu
from codet5p_dataset import CodeT5PDataset


def load_tokenize_data(args):
    with open("data/data.json", 'r') as f:
        data_info = json.load(f)
    assert args.data_name in data_info, "dataset should load in data/data.json"
    if 'hf_hub_url' in data_info[args.data_name]:
        dataset = load_dataset(data_info[args.data_name]['hf_hub_url'])
    else:
        assert "data_dir" in data_info[args.data_name]
        # data_info[data_name]['data_dir'] need train test,valid file
        train_file = glob.glob(os.path.join(data_info[args.data_name]['data_dir'], 'train*'))[0].split('/')[-1]
        valid_file = glob.glob(os.path.join(data_info[args.data_name]['data_dir'], 'valid*'))[0].split('/')[-1]
        test_file = glob.glob(os.path.join(data_info[args.data_name]['data_dir'], 'test*'))[0].split('/')[-1]
        data_files = {
            "train": train_file,
            "validation": valid_file,
            "test": test_file
        }
        dataset = load_dataset(data_info[args.data_name]['data_dir'], data_files=data_files)
    args.eval_data = dataset['validation']['code']
    train_dataset = CodeT5PDataset(dataset['train'], args.tokenizer, is_train=True)
    eval_dataset = CodeT5PDataset(dataset['validation'], args.tokenizer, is_train=True)
    test_dataset = CodeT5PDataset(dataset['test'], args.tokenizer, is_train=False)
    return train_dataset, eval_dataset, test_dataset


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, )
    args.tokenizer = tokenizer
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if args.codet5_b_flag:
        config.pad_token_id = tokenizer.pad_token_id
        # config.decoder_start_token_id = tokenizer.pad_token_id
        config.decoder_start_token_id = tokenizer.bos_token_id
    if args.type == 'full' and args.checkpint_dir:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir, config=config, trust_remote_code=True,
                                                      torch_dtype=torch.float16) if args.fp16 else AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path, config=config, trust_remote_code=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, config=config, trust_remote_code=True,
                                                      torch_dtype=torch.float16) if args.fp16 else AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path, config=config, trust_remote_code=True)
    model.to(args.device)
    print(f"  ==> Loaded model from {args.model_path}, model size {model.num_parameters()}")
    if args.type == 'lora':
        if not args.checkpoint_dir:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                task_type="SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, config)
        else:
            model = PeftModel.from_pretrained(
                model,
                args.checkpoint_dir,
                is_trainable=True
            )
    # don't support yet
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

        model.print_trainable_parameters()
    return model


def get_bleu_socre(ref_file, hyp_file):
    references = []
    df = pd.read_csv(ref_file, header=None)
    fh = df[0].tolist()
    for line in fh:
        refs = [line.strip()]
        references.append([r.split() for r in refs])

    translations = []
    df = pd.read_csv(hyp_file, header=None)
    fh = df[0].tolist()
    for line in fh:
        line = str(line)
        line = line.strip()
        translations.append(line.split())

    assert len(references) == len(translations)
    count = 0
    for i in range(len(references)):
        refs = references[i]  # r is a list of 'list of tokens'
        # print(refs)
        t = translations[i]  # 'list of tokens'
        # print(t)
        for r in refs:
            if r == t:
                count += 1
                break
    acc = round(count / len(translations) * 100, 2)
    bleu_score, _, _, _, _, _ = compute_bleu(references, translations, 4, True)
    bleu_score = round(100 * bleu_score, 2)
    # print('BLEU:\t\t%.2f\nExact Match:\t\t%.2f' % (bleu_score, acc))
    return bleu_score, acc


def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
