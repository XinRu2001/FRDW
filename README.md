# FRDW
This repository is the implementation of “Front-end Replication Dynamic Window (FRDW) for Online Motor Imagery Classification."

## Usage Examples
Here is an example code snippet demonstrating how to use the project:

1. within-subject_train.py
```python
python within-subject_train.py --lr 0.001 --gpu_id '2' --seed 2022 --epoch 180 --bs 64 --train_len 100 --model_num 2 --dataset '001-2014' --classes 2 --person 1 --augmentation overlap --overlap 25 --model_type Transformer --model_save_path /model/try/
```
