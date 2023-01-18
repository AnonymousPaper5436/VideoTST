# VideoTST

This is the official implementation of "Reasoning Video-Language Tasks with Fast-Thinking Sampler and Slow-Thinking Solver". In this paper, we propose a two stage method for videoQA called VideoTST.

![1674041427304](image/README/1674041427304.png)

## Installation

```
pip install -r requirements.txt
```

## Data Preparation

- Download [AGQA 2.0](https://cs.stanford.edu/people/ranjaykrishna/agqa/), and preprocess raw data according to [DeST](https://github.com/shinying/dest)
- The data structure should look like the following:

```
data/
├── kfqa/
    ├── train.json
    ├── val.json
    ├── test.json
    ├── vocab.json
    └── of_feature.h5
```

## Running

Sampler Pre-training

```
bash train.sh
```

Sampler Inferring

```
bash infer.sh
```

## Experiment Results

- [SINGULARITY](https://github.com/jayleicn/singularity)

| Baseline | +Sampler |
| -------- | -------- |
| 51.11    | 53.13    |

- [ALBEF](https://github.com/tsujuifu/pytorch_violet)

| Baseline | +Sampler |
| -------- | -------- |
| 50.68    | 51.73    |

- [VIOLET](https://github.com/salesforce/ALBEF)

| Baseline | +Sampler |
| -------- | -------- |
| 51.03    | 52.59    |
