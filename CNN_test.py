import os, sys
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image #이미지 처리 모듈
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from keras.models import Sequential
from pathlib import Path
import cv2 #오픈소스 컴퓨터 비전 라이브러리. 이미지 및 영상 처리 기능
import random
from MaxPool import MaxPool2

# 데이터 셋 내의 클래스 모두 불러오기
total_classes = os.listdir("C:/Users/dlskd/source/repos/dement/archive/Dataset")
# 각 class별 총 이미지 개수
images_path = Path("C:/Users/dlskd/source/repos/dement/archive/Dataset")
for c in total_classes:
  print(f'* {c}', '=',len(os.listdir(os.path.join(images_path, c))), 'images')


fig,ax = plt.subplots(1,4,figsize=(10,4))
ax = ax.flat
for i,c in enumerate(total_classes):
  img_total_class = list(Path(os.path.join(images_path, c)).glob("*.jpg"))
  img_selected = random.choice(img_total_class)
  img_BGR = cv2.imread(str(img_selected))
  img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
  height,width,channel = img_RGB.shape
  ax[i].imshow(img_RGB)
  ax[i].set_title(f"{img_selected.parent.stem}\nheight:{height}\nwidth:{width}")
  ax[i].axis("off")

fig.tight_layout()
fig.show()

batch_size = 32
img_height = 180
img_width = 180

train = tf.keras.utils.image_dataset_from_directory(
    images_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size = (img_height, img_width),
    batch_size=batch_size)

val = tf.keras.utils.image_dataset_from_directory(
    images_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]


    #yield : return과 유사하지만 yield는 generator 반환
    #generator: 여러 개의 데이터를 미리 만들어 놓지 않고 필요할 때마다 즉석해서 하나씩 만들어낼 수 있는 객체
    #결과값을 나누어서 얻을 수 있기 때문에 성능 측면에서 큰 이점존재

# 3*3 convolutional layer class
class Conv3x3:
    def __init__(self, num_filters): 
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9  # num_filters x 3 x 3 x num_channels

    def iterate_regions(self, image):  # 한 이미지에서 filter가 적용되어야 하는 모든 영역을 생성
        height, width, _ = image.shape

        for i in range(height - 2):  # 필터의 크기가 3 x 3이므로 필터가 이미지에 적용되는 횟수를 맞춰주기 위해 -2 
            for j in range(width - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    '''
    yield : return과 유사하지만 yield는 generator 반환
    generator: 여러 개의 데이터를 미리 만들어 놓지 않고 필요할 때마다 즉석해서 하나씩 만들어낼 수 있는 객체
    결과값을 나누어서 얻을 수 있기 때문에 성능 측면에서 큰 이점존재
    '''
    def forward(self, input):
        self.last_input = input  # Backpropagation 때 사용
        height, width, _ = input.shape
        output = np.zeros((height - 2, width - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3))

        return output

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        input = input.flatten()

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        exp_a = np.exp(totals)
        return exp_a / np.sum(exp_a, axis = 0)
    
conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(89*89*8, 4)

def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Cross entropy loss와 accuracy 계산
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

print("Begin CNN")

loss = 0
num_correct = 0

for i, im in enumerate(val_ds):
    # Forward pass
    _, l, acc = forward(im)
    loss+= l
    num_correct += acc

    # for문이 100번 돌 때마다 실행
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

