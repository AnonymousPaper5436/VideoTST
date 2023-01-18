import os.path as op
import json
import h5py
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseDataset
from .quegen import QuestionGenerator
from .util import sample_frames, warmup_mmap_file




class KFQA(BaseDataset):

    def __init__(self, *args, split, **kwargs):
        # if split == "test": split = "val" # No test split
        assert split in ["train", "val", "test"]
        self.train = split == "train"
        self.split = split
        super().__init__(*args, **kwargs)
        
        data = json.load(open(op.join(self.data_dir, 'kfqa', split+'.json')))
        self.data = list(data.items())
        
        self.ans2id = json.load(open(op.join(self.data_dir, 'kfqa', 'vocab.json')))
        self.ans = sorted(self.ans2id.keys(), key=lambda k: self.ans2id[k])
        
        self.of = np.memmap('data/kfqa/of.feature', dtype='float32',
                            mode='r', shape=(9848, 400, 2, 224, 224))
        self.vid2idx = json.load(open('data/kfqa/vid2idx.json'))
        self.vid2len = json.load(open('data/kfqa/vid2len.json'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, sample = self.data[idx]
        # video = self.get_video(sample['video'])
        video = self.get_of(sample['video'])
        ans = sample['answer']
        label = self.ans2id.get(ans, -1)
        start_label, end_label = self.get_keyframe(video, sample['start'], sample['end'])
        video_len = video.size(0)
        
        return {
            "qid": qid,
            "video": video,
            "text": sample["question"],
            "label": label,
            "start_label": start_label,
            "end_label": end_label,
            "video_length": video_len,
        }
        
    def get_of(self, video_name):
        of = self.of[self.vid2idx[video_name]]
        of = of[:self.vid2len[video_name]]
        of = torch.FloatTensor(of)
        # video = torch.from_numpy(video)
        if of.size(0) > self.max_video_len:
            fid = sample_frames(self.max_video_len, of.size(0), self.sampling)
            of = of[fid]
        return of
    
    def get_video(self, video_name):
        # feat = torch.tensor(self.video_feat['videos/'+video_name][:])
        feat = torch.tensor(self.video_feat[video_name][:])
        if feat.size(0) > self.max_video_len:
            fid = sample_frames(self.max_video_len, feat.size(0), self.sampling)
            feat = feat[fid]
        return feat
        
    def get_keyframe(self, video, start, end):
        video_len = video.size(0)
        start_label, end_label = int(video_len*start)+1, int(video_len*end)+1
        return start_label, end_label

    def encode_text(self, text):
        encoding = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        )
        return encoding

    def encode_choices(self):
        encoding = self.ans_tokenizer(
            self.ans,
            padding=True,
            return_tensors="pt",
        )
        self.ans = encoding

    def collate(self, batch): #, mlm_collator):
        qid = [data["qid"] for data in batch]

        video = pad_sequence([data["video"] for data in batch], batch_first=True)
        video_mask = torch.zeros(video.size(0), video.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            video_mask[i, :data["video"].size(0)] = 1

        questions = [data["text"] for data in batch]
        question_encoding = self.encode_text(questions)

        labels = torch.tensor([data["label"] for data in batch], dtype=torch.long)
        
        start_labels = torch.tensor([data['start_label'] for data in batch], dtype=torch.long)
        end_labels = torch.tensor([data['end_label'] for data in batch], dtype=torch.long)

        video_lengths = torch.tensor([data["video_length"] for data in batch], dtype=torch.long)
        
        return {"qid": qid, "video": video, "video_mask": video_mask,
                "questions": question_encoding, "ans": self.ans,
                "labels": labels, "start_positions": start_labels, "end_positions": end_labels, "video_lengths": video_lengths}
