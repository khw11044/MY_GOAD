# MY_GOAD
---

이 repository에는 ICLR 2020, Liron Bergman 와 Yedid Hoshen의 "[Classification-Based Anomaly Detection for General Data](https://openreview.net/pdf?id=H1lK_lBtvS)"에 제시된 method를 PyTorch로 구현하였습니다.

GOAD는 (Deep Anomaly Detection Using Geometric Transformations)[https://arxiv.org/abs/1805.10917]의 GEOM과 Deep-SVDD의 아이디어를 통합한 방법으로 GEOM은 이미지를 transform하며 one-class를 학습하는 반면 GOAD는 이미지뿐만아니라 tabular data에 대해서도 affine-transform하며 학습하고 anomaly를 detection합니다.

이 repository에는 [GOAD의 공식 github repository](https://github.com/lironber/GOAD)에 있는 코드를 커스텀한 코드가 들어있습니다.

공식 코드는 어떠한 그래프나 표 모델 저장이 없어서 이것들을 추가하였습니다.

또한 저는 AutoEncoder의 reconstruction loss를 추가한 모델 또한 만들었습니다.

# Requirements
---
+ Python 3 +
+ Pytorch 1.0 +
+ Tensorflow 1.8.0 +
+ Keras 2.2.0 +
+ sklearn 0.19.1 +
+ pandas
+ matplotlib

# GOAD

## Training
---

To replicate the results of the paper on the tabular-data:  
> python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=kdd 
python train_ad_tabular.py --n_rots=256 --n_epoch=25 --d_out=128 --ndf=128 --dataset=kddrev
python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=cn7
python train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=thyroid
python train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=arrhythmia 

**dataset**
kdd와 kddrev는 s large-scale cyber intrusion detection datasets
thyroid, arrhythmia : small-scale medical datasets
cn7 : (사출성형기 데이터셋)[https://www.kamp-ai.kr/front/dataset/AiDataDetail.jsp?AI_SEARCH=&page=1&DATASET_SEQ=4&EQUIP_SEL=&FILE_TYPE_SEL=&GUBUN_SEL=&WDATE_SEL=]

To replicate the results of the paper on CIFAR10:  
> python train_ad.py --m=0.1

## demo
---
training을 통해 저장된 model을 불러와 테스트합니다.

> python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=kdd_demo  
python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=cn7_demo

# MY_GOAD

## training 
---
To replicate the results of the paper on the tabular-data:  
> python train_ad_tabular_new_model.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=kdd 
python train_ad_tabular_new_model.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=cn7

GOAD와 상관없는 AutoEncoder 모델 학습 및 demo  
> python train_ad_tabular_AE.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=cn7 

AE는 load한 dataset을 transform 하지 않습니다.
