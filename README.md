# Dementia severity determination : 치매 중증도 판정
## Dataset
Kaggle dataset 사용 : https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset <br/>
MRI image dataset : 128*128 pixels & 6400 images<br/>
classes(labels) : 경미한 치매, 중등도의 치매, 치매가 아닌 이미지, 매우 경미한 치매<br/>
![image](https://github.com/user-attachments/assets/4657d666-245d-4bac-9370-31ea69b9c833)
## CNN tensorflow model
출처 : https://www.kaggle.com/code/abinanthank/alzheimer-s-mri-classification-100-accuracy <br/> 참고해서 모델 조정 <br/>
data augmentation 기법 적용
![image](https://github.com/user-attachments/assets/d88620d3-09fd-4218-8388-f4b002b0d2a6)

### CNN model
- kernel size : 3*3
- number of fileters : 각 컨볼루션 레이어마다 16개 – 32개 - 64개
- number of layers : 3
- activation function : ReLu
- gradient descent optimization alogrithm : adam
- drop out layer : drop out 비율 0.2
### Data split
train dataset : test dataset : validation dataset = 6:2:2
# Results
![image](https://github.com/user-attachments/assets/41d49b30-9881-4c81-b779-944dbbdb78f2)
![image](https://github.com/user-attachments/assets/cd9372c0-54cc-4d47-bf9d-df1b585d2a0b)
![image](https://github.com/user-attachments/assets/2e708bba-45c4-4c2c-a44b-ebc40ffd83e6)
## The Correlation between Batch Size and Learning rate
![image](https://github.com/user-attachments/assets/0191a735-0575-4a3c-ad77-68e72291acd1)
![image](https://github.com/user-attachments/assets/a847c8f1-7483-4032-b3df-09f938861496)

