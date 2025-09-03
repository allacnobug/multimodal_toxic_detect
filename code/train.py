'''
python -W ignore run.py
python block_core.py
find -type d -name 'pymp*' -exec rm -r {} \;
'''

import torch
from argparse import Namespace 
from model import Multimodal_LLM
from dataset import CustomDataset
from iteration import train_model, validate
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel
import warnings
import random
import numpy as np
import torch

warnings.filterwarnings('ignore')

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tasks_bool = {"offensive" : True}
tasks = []
name = "multi3"

for k, v in tasks_bool.items():
    if tasks_bool[k]:
        tasks.append(k)
        # name += k + "_"
        
config = Namespace(
    file_name=name + "_9",
    device=torch.device("cuda:0"),
    tasks = tasks,
    video_encoder="videomae_base",
    audio_encoder="whisper_small",
    video_conv_kernel=36,
    video_conv_stride=24,
    video_conv_padding=0,
    audio_conv_kernel=50,
    audio_conv_stride=23,
    audio_conv_padding=1,
    llm_embed_dim=768,
    llm_output_dim=768,
    # attn_dropout=0.1, # 012
    attn_dropout=0.2,
    is_add_bias_kv=True,
    is_add_zero_attn=True,
    attention_heads=8,
    image_dim=768,
    video_dim=768,
    audio_dim=768,
    video_seq_len=1568,
    audio_seq_len=1500,
    min_mm_seq_len=64, 
    tokenizer_max_len=128,
    add_pooling = False,
    train=True,
    directory = "checkpoints/",
    results_directory = "results/"
)

df_train = pd.read_csv("MultiHateClip/Chinese_data/train_data.csv")
df_val = pd.read_csv("MultiHateClip/Chinese_data/valid_data.csv")
df_test = pd.read_csv("MultiHateClip/Chinese_data/test_data.csv")


num_epochs = 10
batch_size = 2

#roberta 
model_name="/root/toximeme/models/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_name)

special_tokens_dict = {
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
    'pad_token': '[PAD]'  
}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model= BertModel.from_pretrained(model_name)

# tokenizer 添加了新 token，需要扩展模型 embedding
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

print("BOS ID:", tokenizer.bos_token_id)
print("EOS ID:", tokenizer.eos_token_id)
print("PAD ID:", tokenizer.pad_token_id)



model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=model)

train_ds = CustomDataset(dataframe=df_train, train=True, tokenizer=tokenizer)
val_ds = CustomDataset(df_val, train=True, tokenizer=tokenizer)
test_ds = CustomDataset(df_test, train=False, tokenizer=tokenizer)

train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, drop_last=True)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, drop_last=True)

# train
# train_model(model, train_dataloader, val_dataloader, config, num_epochs, "offensive", "f1", devices=None)

# test
checkpoint_path = "checkpoints/multi3_8.pth"

state_dict = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(state_dict)

validate(model, test_dataloader, config)

