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
warnings.filterwarnings("ignore")



# data set 내 클래스 확인
total_classes = os.listdir("C:/Users/dlskd/source/repos/dement/archive/Dataset")
print(total_classes)

# 각 클래스별 이미지 수 확인
images_path = Path("C:/Users/dlskd/source/repos/dement/archive/Dataset")
for c in total_classes:
  print(f'* {c}', '=',len(os.listdir(os.path.join(images_path, c))), 'images')



# =========================
total_size = 6400
batch_size = 32
img_height =128
img_width = 128
# =========================



#data set load
data = tf.keras.utils.image_dataset_from_directory(images_path,
                                                batch_size = batch_size,
                                                image_size=(128, 128),
                                                shuffle=True,
                                                seed=42)
total_batches = len(data)

# Calculate the number of batches for training and validation
train_batches = int(total_batches * 0.8)
val_batches = (total_batches - train_batches) // 2



# Split the dataset into training, validation, and test sets
train_ds = data.take(train_batches)
val_ds = data.skip(train_batches).take(val_batches)
test_ds = data.skip(train_batches + val_batches)
print("total data size: ", len(data))
print("train data size : ", len(train_ds))
print("val data size :", len(val_ds))
print("test data size :", len(test_ds))



#데이터셋 fetch
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)


# data normalization
normalization_layer = layers.Rescaling(1./255) #[0, 255] range의 input을 [0, 1] range로 rescale

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# pizxel값이 [0,1] 내에 있는지 확인
print(np.min(first_image), np.max(first_image))

num_classes = len(total_classes)
print('num classes: ', len(total_classes))



#이미지 데이터 증강
data_augmentation = keras.Sequential(
[
    layers.RandomFlip("horizontal",
                     input_shape=(img_height,
                                 img_width,
                                  3), seed = 42),
    
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


#증가된 이미지 데이터 출력
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(8):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(4,2, i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
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

learning_rate = 0.0001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# gradient descent optimazation algorithm : Adam
# loss 계산 function : Sparse Categorical Crossentropy
model.compile(optimizer=optimizer,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

model.summary()     #모델 개요 출력


# ==============================
epochs = 30    # epoch 값 설정
epochs_range = range(epochs)
# ==============================



# 학습
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")




# test data, validation data에 대한 loss와 accuracy를 그래프로 분석, 출력
plt.figure(figsize=(8,8))
plt.suptitle(f'Model Training Metrics\nBatch Size: {batch_size}, Learning Rate: {learning_rate}', fontsize=13, fontweight='bold')
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
