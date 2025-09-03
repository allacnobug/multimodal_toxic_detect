import torch
from argparse import Namespace 
from model import Multimodal_LLM
from dataset import CustomDataset
from iteration import test
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel
import warnings
import random
import numpy as np


from transformers import logging
logging.set_verbosity_error()
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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
        
config = Namespace(
    file_name=name,
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

df_test = pd.read_csv("MultiHateClip/Chinese_data/test_data.csv")



print("Prepare Tokenizer...")
model_name="/root/toximeme/models/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_name)

special_tokens_dict = {
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
    'pad_token': '[PAD]'  
}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

print("Prepare BertModel...")
model= BertModel.from_pretrained(model_name)

if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

print("Prepare Multimodal_LLM...")
batch_size=1
model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=model)
checkpoint_path = "checkpoints/multi3_model.pth"
state_dict = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(state_dict)

print("Prepare Data...")
test_ds = CustomDataset(df_test[:10], train=False, tokenizer=tokenizer)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, drop_last=True)

print("Testing...")
test(model, test_dataloader, config)

