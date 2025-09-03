"""
Copied from ToxVidLM

"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, WhisperModel
from code.additional_modules import Conv1d_fc, FC_head, Gate_Attention

class Multimodal_LLM(nn.Module):
    
    def __init__(self, batch_size, config, tokenizer, adapter_llm):
        super(Multimodal_LLM, self).__init__()
                
        self.config = config
        
        self.batch_size = batch_size
        
        self.tokenizer = tokenizer
            
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        
        self.output_seq_len = min(self.config.audio_seq_len, self.config.video_seq_len, self.config.min_mm_seq_len)
        
        #model
        self.video_encoder = VideoMAEModel.from_pretrained(config.video_encoder)
        self.audio_encoder =  WhisperModel.from_pretrained(config.audio_encoder).encoder
        # text embed model
        self.adapter_llm = adapter_llm

        self.transform_audio_to_hidden = Conv1d_fc(encoder_embed_dim=self.config.audio_dim,
                                                   llm_embed_dim=self.config.llm_embed_dim,
                                                    kernel_size=self.config.audio_conv_kernel, stride=self.config.audio_conv_stride, padding=self.config.audio_conv_padding)
        self.transform_video_to_hidden = Conv1d_fc(encoder_embed_dim=self.config.video_dim,
                                                   llm_embed_dim=self.config.llm_embed_dim,
                                                    kernel_size=self.config.video_conv_kernel, stride=self.config.video_conv_stride, padding=self.config.video_conv_padding)
    
        self.video_align_attention = nn.MultiheadAttention(self.config.llm_embed_dim, 
                                                             self.config.attention_heads * 2,
                                                             dropout=self.config.attn_dropout,
                                                             add_bias_kv=self.config.is_add_bias_kv,
                                                             add_zero_attn=self.config.is_add_zero_attn)
        self.audio_align_attention = nn.MultiheadAttention(self.config.llm_embed_dim, 
                                                             self.config.attention_heads * 2,
                                                             dropout=self.config.attn_dropout,
                                                             add_bias_kv=self.config.is_add_bias_kv,
                                                             add_zero_attn=self.config.is_add_zero_attn)
      
        self.gate_fusion = Gate_Attention(num_hidden_a = self.config.llm_embed_dim, num_hidden_b = self.config.llm_embed_dim, num_hidden = self.config.llm_embed_dim)
        
        self.offensive_head = FC_head(num_classes=2, hidden_dim=64, llm_embed_dim=self.config.llm_output_dim, add_pooling=self.config.add_pooling)
    
        self.criterion_offensive = nn.CrossEntropyLoss()
        
    def forward(self, inputs):
        #???
        batch_size = self.batch_size
        
        #ids
        bos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device) * self.tokenizer.bos_token_id
        sep = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device) * self.tokenizer.eos_token_id
        eos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device) * self.tokenizer.eos_token_id

        #mask
        attention_mask_bos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device)
        attention_mask_multimodal = torch.ones([batch_size, self.output_seq_len+1], dtype=torch.int64, device=self.config.device)
        attention_mask_eos = torch.zeros([batch_size, 1], dtype=torch.int64, device=self.config.device)
        
        #type_ids
        token_type_ids_bos = torch.zeros([batch_size, 1], dtype=torch.int64, device=self.config.device)
        token_type_ids_multimodal = torch.zeros([batch_size, self.output_seq_len+1], dtype=torch.int64, device=self.config.device)
        token_type_ids_eos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        
        # roberta
        embed_tokens = self.adapter_llm.embeddings.word_embeddings 
        token_embeddings = embed_tokens.weight.unsqueeze(0).repeat(batch_size, 1, 1).transpose(0, 1).contiguous()
        text_embeds = self.adapter_llm.embeddings.word_embeddings(input_ids) #change this as per llm

        #multimodal processing
        video_encoder_out = self.video_encoder(inputs["video"])
        audio_encoder_out = self.audio_encoder(inputs["audio"])
            
        video_encoder_out = self.transform_video_to_hidden(video_encoder_out.last_hidden_state)
        audio_encoder_out = self.transform_audio_to_hidden(audio_encoder_out.last_hidden_state)

        # nn.MultiheadAttention(query, key, value)
        video_encoder_out = self.video_align_attention(video_encoder_out.transpose(0, 1).contiguous(), token_embeddings,
                                                       token_embeddings)[0].transpose(0, 1).contiguous()
        audio_encoder_out = self.audio_align_attention(audio_encoder_out.transpose(0, 1).contiguous(), token_embeddings,
                                                       token_embeddings)[0].transpose(0, 1).contiguous()
        
                
        level_2 = self.gate_fusion(video_encoder_out, audio_encoder_out)
        
        #for roberta
        bos_embeds = self.adapter_llm.embeddings.word_embeddings(bos)
        sep_embeds = self.adapter_llm.embeddings.word_embeddings(sep)
        eos_embeds = self.adapter_llm.embeddings.word_embeddings(eos)
        
        
        text_embeds = torch.cat([bos_embeds, level_2, sep_embeds, text_embeds, eos_embeds], dim=1)

        attention_mask = torch.cat([attention_mask_bos, attention_mask_multimodal, attention_mask, attention_mask_eos],dim=1)

        token_type_ids = torch.cat([token_type_ids_bos, token_type_ids_multimodal, token_type_ids, token_type_ids_eos],dim=1)

        #roberta
        llm_outputs = self.adapter_llm(inputs_embeds=text_embeds, attention_mask=attention_mask).last_hidden_state
                
        outputs = {}
        flag=True
                
        offensive_logits = self.offensive_head(llm_outputs)
        outputs["offensive"] = offensive_logits
        if not flag:
            outputs["loss"] = self.criterion_offensive(offensive_logits, inputs["offensive"])
            flag=True
                
        return outputs