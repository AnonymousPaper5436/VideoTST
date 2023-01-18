# Inference
python run.py with \
    data_root=data \
    num_gpus=1 \
    num_nodes=1 \
    task_finetune_kfqa \
    per_gpu_batchsize=128 \
    test_only=True \
    load_path=YOURPATHHERE 