from datasets import load_dataset
from utils import TARGETS


openorca                    = load_dataset("Open-Orca/OpenOrca", split="train[:500000]")
split                       = openorca.train_test_split(test_size=0.01, seed=42)
openorca_train              = split["train"]
openorca_val                = split["test"]
openorca_train_data         = []
openorca_val_data           = []



for sample in openorca_train:
    conversation = (
        f"<|BOS|><|USER|>{sample['question'].strip()}<|EOS|>"
        f"<|ASSISTANT|>{sample['response'].strip()}<|EOS|><|EOT|>"
    )
    openorca_train_data.append(conversation)

for sample in openorca_val:
    conversation = (
        f"<|BOS|><|USER|>{sample['question'].strip()}<|EOS|>"
        f"<|ASSISTANT|>{sample['response'].strip()}<|EOS|><|EOT|>"
    )
    openorca_val_data.append(conversation)


print("=========================train dat============")
print(openorca_train_data[:2])
print(len(openorca_train_data))
print("=========================val dat============")
print(openorca_val_data[:2])
print(len(openorca_val_data))
