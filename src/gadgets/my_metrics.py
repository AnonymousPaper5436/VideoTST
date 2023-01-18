import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False, topk=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits = logits.detach().to(self.correct.device)
        target = target.detach().to(self.correct.device)

        logits = logits[target!=-100]
        target = target[target!=-100]
        if target.numel() == 0:
            return 1

        if self.topk == 1:
            preds = logits.argmax(dim=-1)
            acc = torch.sum(preds==target)
        else:
            preds = logits.topk(k=self.topk)[1]
            acc = (preds==target.unsqueeze(1)).any(dim=1).sum()

        self.correct += acc
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total
    
class IoU(Metric):
    def __init__(self, dist_sync_on_step=False, topk=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, start_logits, end_logits, start_target, end_target):
        start_logits = start_logits.detach().to(self.correct.device)
        end_logits = end_logits.detach().to(self.correct.device)
        start_target = start_target.detach().to(self.correct.device)
        end_target = end_target.detach().to(self.correct.device)

        start_logits = start_logits[start_target!=-100]
        satrt_target = start_target[start_target!=-100]
        end_logits = end_logits[end_target!=-100]
        end_target = end_target[end_target!=-100]
        
        if start_target.numel() == 0 or end_target.numel() == 0:
            return 1
        
        start_preds = start_logits.argmax(dim=1)
        end_preds = end_logits.argmax(dim=1)
        
        for i in range(start_target.size(0)):
            if start_preds[i] == start_target[i] and end_preds[i] == end_target[i]:
                self.correct += 1.
            elif start_preds[i] >= end_target[i] or end_preds[i] <= start_target[i]:
                self.correct += 0.
            else:
                ll = min(start_preds[i], start_target[i])
                lr = max(start_preds[i], start_target[i])
                rr = max(end_preds[i], end_target[i])
                rl = min(end_preds[i], end_target[i])
                self.correct += (rl-lr)/(rr-ll)
            

        self.total += start_target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total
