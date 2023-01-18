import os
import glob
import json
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn as dist_nn

from .dist_utils import all_gather, get_rank, get_world_size


def gather(tensor):
    world_size = get_world_size()
    device = tensor.device

    local_size = torch.tensor(tensor.size(0), device=device)
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)

    M = max(sizes)
    if local_size < M:
        tensor = torch.cat([tensor, torch.zeros(M-local_size, tensor.size(1), device=device)])
    tensors = dist_nn.all_gather(tensor)
    tensors = [tensor[:s] for tensor, s in zip(tensors, sizes)]
    return torch.cat(tensors)



def compute_kfqa(pl_module, batch):
    ques_hidden = pl_module.encode_question(batch)
    vid_hidden, vid_feat = pl_module.encode_video(batch, ques_hidden, batch["questions"].attention_mask, return_hidden=True)
    
    cls_feat = vid_feat
    ans_feat = pl_module.ans_encoder(batch["ans"])
    
    # video qa    
    logits = cls_feat @ ans_feat.T
    loss = F.cross_entropy(logits, batch['labels'], ignore_index=-1)
    
    # video mrc
    kfqa_logits = pl_module.mrc_head(vid_hidden)
    start_logits, end_logits = kfqa_logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    ignored_index = start_logits.size(1)
    start_positions = batch['start_positions'].clamp(0, ignored_index)
    end_positions = batch['end_positions'].clamp(0, ignored_index)
    start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
    end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
    mrc_loss = (start_loss + end_loss) / 2
    
    loss = 0.1*loss + 0.9*mrc_loss

    ret = {
        "kfqa_mrc_loss": mrc_loss,
        "kfqa_start_logits": start_logits,
        "kfqa_end_logits": end_logits,
        "kfqa_loss": loss,
        "kfqa_logits": logits,
        "kfqa_lables": batch['labels'],
    }
    
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_kfqa_loss")(loss)
    score = getattr(pl_module, f"{phase}_kfqa_accuracy")(logits, batch["labels"])
    start_loss = getattr(pl_module, f"{phase}_start_loss")(start_loss)
    start_score = getattr(pl_module, f"{phase}_start_accuracy")(start_logits, batch["start_positions"])
    end_loss = getattr(pl_module, f"{phase}_end_loss")(end_loss)
    end_score = getattr(pl_module, f"{phase}_end_accuracy")(end_logits, batch["end_positions"])
    iou_score = getattr(pl_module, f"{phase}_iou")(start_logits, end_logits, batch["start_positions"], batch["end_positions"])
    
    B = logits.size(0)
    pl_module.log(f"kfqa/{phase}/loss", loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"kfqa/{phase}/accuracy", score, batch_size=B, rank_zero_only=True)
    pl_module.log(f"kfqa/{phase}/start_loss", start_loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"kfqa/{phase}/start_accuracy", start_score, batch_size=B, rank_zero_only=True)
    pl_module.log(f"kfqa/{phase}/end_loss", end_loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"kfqa/{phase}/end_accuracy", end_score, batch_size=B, rank_zero_only=True)
    pl_module.log(f"kfqa/{phase}/iou_score", iou_score, batch_size=B, rank_zero_only=True)
    
    return ret



def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def kfqa_test_step(pl_module, batch, output):
    start_logits = output["kfqa_start_logits"]
    end_logits = output["kfqa_end_logits"]
    video_lengths = batch['video_lengths']
    
    # predict start & end
    start_prob = nn.Softmax(dim=1)(start_logits)
    end_prob = nn.Softmax(dim=1)(end_logits)
    start_preds = list()
    end_preds = list()
    
    
    # top k
    for ii in range(start_logits.size(0)):
        start_score = start_prob[ii]
        end_score = end_prob[ii]
        video_length = video_lengths[ii]
        best_score = 0
        k = min(10, video_length)
        start_pred_topk = torch.topk(start_score[1:video_length+1], k, dim=0).indices
        end_pred_topk = torch.topk(end_score[1:video_length+1], k, dim=0).indices
        start_pred = 0
        end_pred = video_length-1
        for start_index in start_pred_topk:
            for end_index in end_pred_topk:
                if start_index >= end_index: continue
                if start_score[start_index+1] + end_score[end_index+1] > best_score:
                    best_score = start_score[start_index+1] + end_score[end_index+1]
                    start_pred = start_index
                    end_pred = end_index
        start_preds.append(start_pred.cpu().item())
        end_preds.append(end_pred.cpu().item())
    
    return {"start_preds":start_preds, "end_preds":end_preds, "start_postions":batch["start_positions"].cpu(), "end_positions": batch["end_positions"].cpu(),\
        "qids":batch['qid'], "video_lengths": batch['video_lengths']}
    
def kfqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, start_preds, end_preds, video_lengths = list(), list(), list(), list()
    
    for out in outs:
        qids += out["qids"]
        start_preds += out["start_preds"]
        end_preds += out["end_preds"]
        video_lengths += out['video_lengths'].tolist()

    rets = [{"qid": qid, "start_pred": start_pred/video_length, "end_pred": end_pred/video_length} \
            for qid, start_pred, end_pred, video_length in zip(qids, start_preds, end_preds, video_lengths)]
    with open(f"kfqa_tmp_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("kfqa_tmp_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/kfqa_test_by_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"kfqa_tmp_{rank}.json")
