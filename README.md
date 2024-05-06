# MY_GOAD
---

이 repository에는 ICLR 2020, Liron Bergman 와 Yedid Hoshen의 "[Classification-Based Anomaly Detection for General Data](https://openreview.net/pdf?id=H1lK_lBtvS)"에 제시된 method를 PyTorch로 구현하였습니다.

GOAD는 (Deep Anomaly Detection Using Geometric Transformations)[https://arxiv.org/abs/1805.10917]의 GEOM과 Deep-SVDD의 아이디어를 통합한 방법으로 GEOM은 이미지를 transform하며 one-class를 학습하는 반면 GOAD는 이미지뿐만아니라 tabular data에 대해서도 affine-transform하며 학습하고 anomaly를 detection합니다.

이 repository에는 [GOAD의 공식 github repository](https://github.com/lironber/GOAD)에 있는 코드를 커스텀한 코드가 들어있습니다.

공식 코드는 어떠한 그래프나 표 모델 저장이 없어서 이것들을 추가하였습니다.

또한 저는 AutoEncoder의 reconstruction loss를 추가한 모델 또한 만들었습니다.

설명은 [블로그](https://khw11044.github.io/project/2021-06-15-My_GOAD/)를 참조.

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

Cifa10과 각종 tabular 데이터셋에 대해 불균형 데이터셋 또는 Anomaly Detection을 수행할 수 있습니다. 

기존 GOAD보다 훨씬 빨리 훈련을 수행하며 시각 자료를 제공합니다. 

훈련과 테스트에 최적화되게 코드를 개선하였습니다.

## Training
---

Tabular dataset에 대해 

train_ad_tabular 또는 train_AE_tabular 또는 train_AE_tabular_compound 수행하면 됩니다.

~~~
python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=kdd 
python train_ad_tabular.py --n_rots=256 --n_epoch=25 --d_out=128 --ndf=128 --dataset=kddrev
python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=cn7
python train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=thyroid
python train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=arrhythmia 
~~~

**dataset**
kdd와 kddrev는 s large-scale cyber intrusion detection datasets
thyroid, arrhythmia : small-scale medical datasets
cn7 : (사출성형기 데이터셋)[https://www.kamp-ai.kr/front/dataset/AiDataDetail.jsp?AI_SEARCH=&page=1&DATASET_SEQ=4&EQUIP_SEL=&FILE_TYPE_SEL=&GUBUN_SEL=&WDATE_SEL=]

CIFAR10 훈련에 대해 

train_ad_image 수행하면 됩니다.


Cn7 데이터셋에 대해 

![24_pr_curve_graph](https://github.com/khw11044/DeepSVDD-Pytorch-Mine/assets/51473705/d96df66f-f728-4633-b138-b7080fb3d59f)

![24_roc_curve_graph](https://github.com/khw11044/DeepSVDD-Pytorch-Mine/assets/51473705/9ec57651-4454-42a0-bb98-1432a2175228)

![24_val_result_vis](https://github.com/khw11044/DeepSVDD-Pytorch-Mine/assets/51473705/2ef591aa-e9da-43e7-af7e-1b6662408048)

![AE_error_graph](https://github.com/khw11044/DeepSVDD-Pytorch-Mine/assets/51473705/d7bec100-1748-432e-95f7-dbeb01165481)
