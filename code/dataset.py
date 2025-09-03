"""
返回一个Dataset

"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import return_audio_tensor,return_video_tensor, prepare_batch
from ast import literal_eval
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Normalize,
)
            

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, train):
        self.dataframe = dataframe
        self.train = train
        self.tokenizer = tokenizer
        
        # whether data augmention
        if self.train:
            self.video_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop((224,224)),
                    RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.video_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # title,description,transcript
        cols = ["title", "description", "transcript"]
        dialogue = " ".join([str(self.dataframe.loc[idx, c]) if pd.notna(self.dataframe.loc[idx, c]) else "" for c in cols])
        # offensive_labels = torch.tensor(literal_eval(self.dataframe["Majority_Voting"].iloc[idx]), device="cpu")
        val = self.dataframe["Majority_Voting"].iloc[idx]
        offensive_labels = torch.tensor(val, device="cpu")
        video = self.dataframe['video_path'].iloc[idx]
        audio = self.dataframe['audio_path'].iloc[idx]
        
        audio = return_audio_tensor(audio)
        video = return_video_tensor(video)
                
        video = self.video_transform(video)
        
        sample = {
            'dialogue' : dialogue,
            'offensive' : offensive_labels,
            'video': video,
            'audio': audio,
        }
        
        sample = prepare_batch(batch=sample, tokenizer=self.tokenizer, train=self.train)

        return sample