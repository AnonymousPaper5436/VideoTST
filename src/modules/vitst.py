import copy
from functools import partial

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertModel
from . import heads
from . import objectives
from . import utils


class ViTST(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.tasks = config["loss_names"]
        assert config["load_path"], "no checkpoints"

        self.visual_encoder = VisionTransformer(
            img_size=config["image_size"], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig(**config["bert_config"])
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", config=bert_config,
                                                      add_pooling_layer=False)


        # ==================== Checkpoint for Pretrain =================== #

        if self.tasks["kfqa"] and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    new_key = key.replace('bert.', '')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            msg = self.load_state_dict(state_dict, strict=False)
            print(">>> Load checkpoint for pretrain from", config["load_path"])
            utils.parse_loading_msg(msg)


        # ==================== Video Encoding =================== #
        
        if self.tasks["kfqa"]:
            D = self.text_encoder.config.hidden_size
            with torch.no_grad():
                pos_embed = self.text_encoder.embeddings.position_embeddings.weight.data[:config["max_pos_len"]+2]
            self.temporal_embedding = TemporalOFEmbedding(config["input_video_embed_size"],
                    D, pos_embed, config["max_pos_len"], config["drop_rate"])

            self.temporal_encoder = copy.deepcopy(self.text_encoder)
            del self.temporal_encoder.embeddings

        # ==================== Pretarin-specific Modules =================== #

        D = self.text_encoder.config.hidden_size

        if self.tasks["kfqa"]:
            self.vision_proj = nn.Linear(D, D//3)
            self.vision_proj.apply(objectives.init_weights)

            self.text_proj = nn.Linear(D, D//3)
            self.text_proj.apply(objectives.init_weights)

            self.tau = nn.Parameter(torch.tensor(0.07))

            self.ans_encoder = AnsEncoder(config["hidden_size"]//2)
            self.video_head = nn.Sequential(
                    nn.Linear(D, D),
                    nn.ReLU(),
                    nn.Linear(D, D//2))
            self.video_head.apply(objectives.init_weights)
            
            self.mrc_head = nn.Sequential(
                nn.Linear(D, D),
                nn.ReLU(),
                nn.Linear(D, 2)
            )
            # self.mrc_head = nn.Linear(D, 2)
            self.mrc_head.apply(objectives.init_weights)

        # ====================  Inference =================== #

        if config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict)
            print(">>> Load checkpoint for inference from", config["load_path"])

        self.freeze_weights()
        self.set_forward(config)
        utils.set_metrics(self)

    def freeze_weights(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

    def encode_question(self, batch):
        text_output = self.text_encoder(batch["questions"].input_ids,
                                        attention_mask=batch["questions"].attention_mask,
                                        mode="text")
        return text_output["last_hidden_state"]

    def encode_video(self, batch, text_hidden_states=None, text_mask=None, return_hidden=True):
        video_embed = self.temporal_embedding(batch)
        video_embed = self.temporal_encoder(encoder_embeds=video_embed,
                                            attention_mask=batch["video_mask"],
                                            mode="text")[0]
        if text_hidden_states is None:
            return video_embed # B, seq_len, hdsz

        video_output = self.temporal_encoder(encoder_embeds=video_embed,
                                             attention_mask=batch["video_mask"],
                                             encoder_hidden_states=text_hidden_states,
                                             encoder_attention_mask=text_mask,
                                             mode="fusion")
        video_feat = video_output[0][:, 0]
        video_feat = self.video_head(video_feat)
        fused_feat = video_output[0]

        if return_hidden:
            return fused_feat, video_feat
        return video_feat


    def set_forward(self, config):
        if self.tasks["kfqa"]:
            self.forward_task = objectives.compute_kfqa
            self.test_forward = objectives.kfqa_test_step
            self.test_wrapup = objectives.kfqa_test_wrapup
        else:
            raise ValueError("loss is not set")

    def forward(self, batch):
        return self.forward_task(self, batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def validation_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        return self.test_forward(self, batch, output)

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        self.test_wrapup(outs, model_name)
        utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return utils.set_schedule(self)

class TemporalOFEmbedding(nn.Module):

    def __init__(self, input_size, embed_size, position_embedding_data, max_pos_len=100, dropout=0):
        super().__init__()
        

        self.projection = nn.Conv2d(2, 1, kernel_size=8, stride=8)
        input_size = (224 // 8) ** 2
        
        
        self.fc = nn.Linear(input_size, embed_size)
        self.bos = nn.Parameter(torch.empty(embed_size))
        self.eos = nn.Parameter(torch.empty(embed_size))

        # Frame positional embedding
        self.register_buffer("position_ids", torch.arange(max_pos_len+2).expand(1, -1))
        self.frame_pos_embed = nn.Embedding(max_pos_len+2, embed_size, _weight=position_embedding_data)
        self.ln = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.projection.apply(objectives.init_weights)
        self.fc.apply(objectives.init_weights)
        # self.proj.apply(objectives.init_weights)
        nn.init.trunc_normal_(self.bos, mean=0, std=0.02)
        nn.init.trunc_normal_(self.eos, mean=0, std=0.02)
        self.ln.apply(objectives.init_weights)

    def forward(self, batch):
        B, L, C, H, W = batch['video'].size()
        video_embed = self.projection(batch['video'].view(-1, C, H, W)) # -> B*L, 1024, 1, 1 / B*L, 1, 28, 28
        video_embed = video_embed.flatten(1).view(B, L, -1) # -> B, L, 747
        video_embed = self.fc(video_embed) # 747 -> 768
        # video_embed = self.proj(video_embed)
        B, S, D = video_embed.size()
        video_embed = torch.cat([self.bos.expand(B, 1, -1), video_embed,
                                  torch.zeros(B, 1, D, device=video_embed.device)], dim=1)
        ends = batch["video_mask"].sum(dim=1) - 1
        video_embed[torch.arange(B), ends] = self.eos

        pos_ids = self.position_ids[:, :video_embed.size(1)]
        video_embed += self.frame_pos_embed(pos_ids)
        video_embed = self.ln(video_embed)
        video_embed = self.dropout(video_embed)

        return video_embed


class AnsEncoder(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        D = self.lm.config.dim
        self.pooler = nn.Sequential(
            heads.Pooler(D),
            nn.Linear(D, output_size),
        )
        self.pooler.apply(objectives.init_weights)

        for param in self.lm.embeddings.parameters():
            param.requires_grad = False

    def forward(self, encoding):
        feat = self.lm(**encoding)[0] # last hidden state
        return self.pooler(feat)

