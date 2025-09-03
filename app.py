from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import tempfile
import os
from code.extract_npy import extract_audio_from_mp4
from argparse import Namespace 
import torch
from transformers import BertTokenizer, BertModel, AutoImageProcessor, AutoProcessor
from code.model import Multimodal_LLM
from code.utils import return_audio_tensor,return_video_tensor, prepare_batch_test,process_video
from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
)
from transformers import logging
logging.set_verbosity_error()
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


app = FastAPI()
# ===== 模型初始化（只执行一次） =====
     
config = Namespace(
    file_name="multi3",
    device=torch.device("cuda:0"),
    tasks = ["offensive"],
    video_encoder="models/videomae_base", # video encoder path
    audio_encoder="models/whisper_small", # audio encoder path
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
    train=False,
    directory = "checkpoints/",
    results_directory = "training_detail/"
)
# Roberta path
model_name="models/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_name)
special_tokens_dict = {
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
    'pad_token': '[PAD]'  
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model= BertModel.from_pretrained(model_name)

if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

batch_size=1
model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=model)
checkpoint_path = "checkpoints/multi3_model.pth"
state_dict = torch.load(checkpoint_path, map_location=config.device,weights_only=True)
model.load_state_dict(state_dict)
model = model.to(config.device)
model.eval()

audio_processor = AutoProcessor.from_pretrained("models/whisper_small")
video_processor = AutoImageProcessor.from_pretrained("models/videomae_base", use_fast=True)

# ========== 多模态接口 ==========
# 需要上传视频，文字（视频标题，介绍，字幕。。。）
@app.post("/predict_multimodal")
async def predict_multimodal(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    
    # 1. save video as tmp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    # 2. extract_audio_from_mp4
    audio_path = video_path.replace(".mp4", ".npy")
    extract_audio_from_mp4(video_path, audio_path)

    # 3. process video audio
    process_video(video_path)
    video_transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # 4. video audio embedding 
    audio = return_audio_tensor(audio_path, audio_processor)
    video = return_video_tensor(video_path, video_processor)
    video = video_transform(video)
    #add batch dim
    audio = audio.unsqueeze(0)
    video = video.unsqueeze(0)
    sample = {
            'dialogue' : text,
            'video': video,
            'audio': audio,
        }
    sample = prepare_batch_test(batch=sample, tokenizer=tokenizer)
    # tensor to device
    sample = {k: (v.to(config.device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    
    with torch.no_grad():
        outputs = model(sample)
    predictions = torch.argmax(outputs["offensive"], dim=1).detach().cpu().numpy()

    # remove tmp
    os.remove(video_path)
    os.remove(audio_path)

    predictions = int(predictions[0])
    if predictions == 0:
        result = 'Normal'
    else:
        result = 'Offensive'
    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)