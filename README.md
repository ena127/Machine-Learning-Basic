# Dementia severity determination : 치매 중증도 판정
## Dataset
Kaggle dataset 사용 : https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset <br/>
MRI image dataset : 128*128 pixels & 6400 images<br/>
classes(labels) : 경미한 치매, 중등도의 치매, 치매가 아닌 이미지, 매우 경미한 치매<br/>
## CNN tensorflow model
출처 : https://www.kaggle.com/code/abinanthank/alzheimer-s-mri-classification-100-accuracy <br/> 참고해서 모델 조정 <br/>
data augmentation 기법 적용
### CNN model
- kernel size : 3*3
- number of fileters : 각 컨볼루션 레이어마다 16개 – 32개 - 64개
- number of layers : 3
- activation function : ReLu
- gradient descent optimization alogrithm : adam
- drop out layer : drop out 비율 0.2
### Data split
train dataset : test dataset : validation dataset = 6:2:2
