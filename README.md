# FRDW
This repository is the implementation of “Front-end Replication Dynamic Window (FRDW) for Online Motor Imagery Classification."

## Usage Examples
Here is an example code snippet demonstrating how to use the project:

1. within-subject_train: for finding the best parameters and training the models
```python
python within-subject_train.py --lr 0.001 --gpu_id '2' --seed 2022 --epoch 180 --bs 64 --train_len 100 --model_num 2 --dataset '001-2014' --classes 2 --person 1 --augmentation overlap --overlap 25 --model_type Transformer --model_save_path /model/try/
```
2. cross-subject_train: for finding the best parameters and training the models
```python
python cross-subject_train.py --lr 0.001 --gpu_id '2' --seed 2022 --epoch 180 --bs 64 --train_len 100 --model_num 2 --dataset '004-2014' --classes 2  --person 1 --augmentation overlap --overlap 25 --model_type Transformer --model_save_path /model/try/ --if_EA True
```
3. FW: fixed window (FW) testing
4. FRDW: FRDW testing
 ```python
python FW.py --seed 2022 --model_num 2 --train_length 100 --dataset '001-2014' --classes 4 --person 1 --model_type 'EEGNet' --model_save_path /model/EEGNet-001-2014-4/within_overlap/ --gpu_id '3'
python FRDW.py --seed 2022 --model_num 2 --train_length 100 --dataset '001-2014' --classes 4 --person 1 --model_type 'EEGNet' --model_save_path /model/EEGNet-001-2014-4/within_overlap/ --gpu_id '3'
```
5. FW+EA: fixed window (FW) with EA testing
6. FRDW+EA: FRDW with EA testing
```python
python FW+EA.py --seed 2022 --model_num 2 --train_length 100 --dataset '001-2014' --classes 4 --person 1 --model_type 'EEGNet' --model_save_path /model/EEGNet-001-2014-4/cross_overlap/ --gpu_id '3' --modelEA_save_path /model/EEGNet-001-2014-4/cross_overlap_EA/
python FRDW+EA.py --seed 2022 --model_num 2 --train_length 100 --dataset '001-2014' --classes 4 --person 1 --model_type 'EEGNet' --model_save_path /model/EEGNet-001-2014-4/cross_overlap/ --gpu_id '3' --modelEA_save_path /model/EEGNet-001-2014-4/cross_overlap_EA/
```

## Dataset

You can find the dataset at the following locations:
   http://www.bnci-horizon-2020.eu/database/data-sets
   or 
   https://www.bbci.de/competition/iv/
.We have provided data processing code for both .gdf and .mat formats.


   
