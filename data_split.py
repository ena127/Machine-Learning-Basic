
import os
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from keras.models import Sequential
from pathlib import Path
import random
import cv2
import warnings
from keras.callbacks import TensorBoard
warnings.filterwarnings("ignore")

# data set 내 클래스
total_classes = os.listdir("C:/Users/dlskd/source/repos/dement/archive/Dataset")
print(total_classes)

# 각 클래스별 이미지 수
images_path = Path("C:/Users/dlskd/source/repos/dement/archive/Dataset")
for c in total_classes:
  print(f'* {c}', '=',len(os.listdir(os.path.join(images_path, c))), 'images')


data = tf.keras.utils.image_dataset_from_directory(images_path,
                                                batch_size = 32,
                                                image_size=(128, 128),
                                                shuffle=True,
                                                seed=42,)
class_names = data.class_names
alz_dict = {index: img for index, img in enumerate(data.class_names)}
class Process:
    def __init__(self, data):
        self.data = data.map(lambda x, y: (x/255, y))

    def create_new_batch(self):
        self.batch = self.data.as_numpy_iterator().next()
        text = "Min and max pixel values in the batch ->"
        print(text, self.batch[0].min(), "&", self.batch[0].max())
        
    def show_batch_images(self, number_of_images=5):
        fig, ax = plt.subplots(ncols=number_of_images, figsize=(20,20), facecolor="gray")
        fig.suptitle("Brain MRI (Alzheimer) Samples in the Batch", color="yellow",fontsize=18, fontweight='bold', y=0.6)
        for idx, img in enumerate(self.batch[0][:number_of_images]):
            ax[idx].imshow(img)
            class_no = self.batch[1][idx]
            ax[idx].set_title(alz_dict[class_no], color="aqua")
            ax[idx].set_xticklabels([])
            ax[idx].set_yticklabels([])
    
    def train_test_val_split(self, train_size, val_size, test_size):

        train = int(len(self.data)*train_size)
        test = int(len(self.data)*test_size)
        val = int(len(self.data)*val_size)
        
        train_data = self.data.take(train)
        val_data = self.data.skip(train).take(val)
        test_data = self.data.skip(train+val).take(test)

        return train_data, val_data, test_data

process = Process(data)
process.create_new_batch()
process.show_batch_images(number_of_images=5)
train_data, val_data, test_data= process.train_test_val_split(train_size=0.8, val_size=0.1, test_size=0.1)

#데이터셋 mapping
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_data.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_data

'''
dataset.cache()에 대한 설명

버퍼링된 프리페치를 사용하여 I/O를 차단하지 않고 디스크에서 데이터를 생성할 수 있도록 함
Dataset.cache()는 첫 epoch 동안 디스크에서 이미지를 로드한 후 이미지를 메모리에 유지
-> 모델을 훈련하는 동안 dataset이 병목 상태가 되지 않음
'''

normalization_layer = layers.Rescaling(1./255) #[0, 255] range의 input을 [0, 1] range로 rescale

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Notice the pixel values are now in '[0,1]'
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

img_height = 128
img_width = 128

#이미지 데이터 증강
data_augmentation = keras.Sequential(
[
    layers.RandomFlip("horizontal",
                     input_shape=(img_height,
                                 img_width,
                                  3)),
    
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
'''
#증가된 이미지 데이터 출력
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(8):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(4,2, i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
'''
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    augmented_images = data_augmentation(images)
    print("Shape of augmented_images:", augmented_images.shape)  # Add this line to check the shape
    for i in range(8):
        ax = plt.subplot(4, 2, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))  # Modify this line
        plt.axis("off")


#증강된 데이터에 대한 sequential model.
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes)
])

model.compile(optimizer="adam",
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

model.summary()
epochs = 10


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest loss: {test_loss:.4f}")
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")


#accuracy를 그래프로 분석, 출력
plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}
plt.show()

model.save("alzeihmer_model.h5")