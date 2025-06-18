# HF-ReID-DML
The official repository for Revisiting CNN for Hardware-Friendly Person Re-Identification and Deep Metric Learning.

## Prepare Datasets
Download the person datasets, vehicle datasets, and fine-grained Visual Categorization/Retrieval datasets.

Then unzip them and rename them under your "dataset_root" directory like
```bash
dataset_root
├── Market-1501-v15.09.15
├── DukeMTMC-reID
├── MSMT17
├── cuhk03-np
├── VeRi
├── VehicleID_V1.0
├── CARS
├── CUB_200_2011
└── Stanford_Online_Products
```

## Training
We prepared the ImageNet Pretrained RegNet backbone in "./pretrain".

### Train on Occluded_Duke
```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.956057 top5:0.986639 top10:0.991983 mAP:0.885133

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.903501 top5:0.951975 top10:0.963645 mAP:0.801830

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 32 --lr 5.0e-2 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.828030 top5:0.906253 top10:0.928810 mAP:0.624510

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 32 --lr 1.0e-1 --dataset npdetected --gpus 0 --epochs 5,185 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.812857 top5:0.921429 top10:0.948571 mAP:0.772432

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 32 --lr 1.0e-1 --dataset nplabeled --gpus 0 --epochs 5,185 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.835000 top5:0.931429 top10:0.957857 mAP:0.799258

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 256 --img-width 256 --batch-size 32 --lr 5.0e-2 --dataset veri776 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.969011 top5:0.985697 top10:0.992849 mAP:0.810786

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 256 --img-width 256 --batch-size 256 --lr 2.0e-1 --dataset vehicleid --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.872475 top5:0.978395 top10:0.990515 mAP:0.895659

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 224 --img-width 224 --batch-size 24 --lr 2.0e-3 --dataset cub200 --gpus 0 --epochs 5,55 --instance-num 6 --erasing 0.2 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.650743 Recall@2:0.758103 Recall@4:0.843349 Recall@8:0.902768 NMI:0.688525

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 224 --img-width 224 --batch-size 24 --lr 1.0e-2 --dataset car196 --gpus 0 --epochs 5,55 --instance-num 6 --erasing 0.2 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.907145 Recall@2:0.949084 Recall@4:0.969130 Recall@8:0.981798 NMI:0.787291

```bash
python train.py --net e-96_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.818849 Recall@10:0.921242 NMI:0.910172


```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.958729 top5:0.986342 top10:0.990796 mAP:0.892913

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.912926 top5:0.959605 top10:0.969031 mAP:0.813602

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 32 --lr 5.0e-2 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.836950 top5:0.912171 top10:0.934214 mAP:0.638435

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 32 --lr 1.0e-1 --dataset npdetected --gpus 0 --epochs 5,185 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.821714 top5:0.926429 top10:0.958571 mAP:0.782914

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 384 --img-width 128 --batch-size 32 --lr 1.0e-1 --dataset nplabeled --gpus 0 --epochs 5,185 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.851429 top5:0.939286 top10:0.970714 mAP:0.820612

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 256 --img-width 256 --batch-size 32 --lr 5.0e-2 --dataset veri776 --gpus 5 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.970799 top5:0.986889 top10:0.993445 mAP:0.813654

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 256 --img-width 256 --batch-size 256 --lr 2.0e-1 --dataset vehicleid --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
top1:0.879501 top5:0.979975 top10:0.991920 mAP:0.901028

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 224 --img-width 224 --batch-size 24 --lr 2.0e-3 --dataset cub200 --gpus 0 --epochs 5,55 --instance-num 6 --erasing 0.2 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.660196 Recall@2:0.760972 Recall@4:0.848413 Recall@8:0.914922 NMI:0.698835

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 224 --img-width 224 --batch-size 24 --lr 1.0e-2 --dataset car196 --gpus 0 --epochs 5,55 --instance-num 6 --erasing 0.2 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.910220 Recall@2:0.952404 Recall@4:0.972943 Recall@8:0.982905 NMI:0.797155

```bash
python train.py --net e-128_r-1.00_l-3-8-2_gw-4 --decoder avg-2_max-2 --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.4 --triplet-weight 1.0 --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
Recall@1:0.828931 Recall@10:0.928102 NMI:0.913889
```
