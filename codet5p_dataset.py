import torch
from torch.utils.data import Dataset


class CodeT5PDataset(Dataset):
    def __init__(self, dataset, tokenizer, is_train=True):
        super(CodeT5PDataset, self).__init__()
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["nl", "code"],
            num_proc=8,

        )

    def preprocess_function(self, dataset):
        source = [' '.join(ex.split()) for ex in dataset["nl"]]
        if self.is_train:
            target = [' '.join(ex.split()) for ex in dataset["code"]]
        else:
            target = ["" for _ in range(len(dataset['nl']))]
        model_inputs = self.tokenizer(
            source,
            max_length=150,
            padding="max_length",
            truncation=True
        )
        labels = self.tokenizer(
            target,
            max_length=150,
            padding="max_length",
            truncation=True
        )
        return {
            "all_source_ids": model_inputs['input_ids'],
            "all_source_mask": model_inputs['attention_mask'],
            "all_target_ids": labels['input_ids'],
            "all_target_mask": labels['attention_mask']
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx]['all_source_ids']),
            torch.tensor(self.data[idx]['all_source_mask']),
            torch.tensor(self.data[idx]['all_target_ids']),
            torch.tensor(self.data[idx]['all_target_mask'])
        )
