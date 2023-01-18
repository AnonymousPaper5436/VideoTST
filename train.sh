# Training
python run.py with \
    data_root=data \
    num_gpus=1 \
    num_nodes=1 \
    task_finetune_kfqa \
    per_gpu_batchsize=128 \
    load_path=ckpts/trm.ckpt
