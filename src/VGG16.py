#!/usr/bin/env python
# coding: utf-8

# - 太田満久、須藤広大、大澤匠雅、小田大輔、現場で使える！ TensorFlow 開発入門 Keras による深層学習モデル構築手法、翔泳社 2018.
# - 6章 学習済みモデルの活用

# In[1]:


from datetime import datetime
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from utils import load_random_imgs, show_test_samples
import json
import math
import numpy as np
import os
import pickle


# In[2]:


model = VGG16()


# In[3]:


model.summary()


# In[4]:


img_ibo = load_img("../data/input/ibo.jpg", target_size=(224,224))
img_lion = load_img("../data/input/lion.jpg", target_size=(224,224))
img_cloud = load_img("../data/input/cloud.jpg", target_size=(224,224))
img_ibo


# In[5]:


arr_ibo = img_to_array(img_ibo)
arr_lion = img_to_array(img_lion)
arr_cloud = img_to_array(img_cloud)
arr_ibo = preprocess_input(arr_ibo)
arr_lion = preprocess_input(arr_lion)
arr_cloud = preprocess_input(arr_cloud)
arr_input = np.stack([arr_ibo, arr_lion, arr_cloud])
arr_input.shape
# => (3, 224, 224, 3) ... 画像枚数, 224x224, 3ch


# In[6]:


probs = model.predict(arr_input)
print("shape of probs:", probs.shape) # => shape of probs: (3, 1000)  3枚の画像、1000クラスの確率
results = decode_predictions(probs)
print(results[2]) # => 3枚目の画像のクラス確率


# In[7]:


img_ibo


# In[8]:


# 転移学習用のモデル


# In[9]:


# 転移学習用に VGG16 を呼び出す
# 出力層の1000クラス分類は行わないため、 include_top=Flase とする
# input_shape=(224, 224, 3) は自由に変えることができる
vgg16_ft = VGG16(include_top=False, input_shape=(224, 224, 3))
vgg16_ft.summary()


# In[10]:


# モデルの編集
def build_transfer_model(vgg16):
    model = Sequential(vgg16.layers)
    for layer in model.layers[:15]:
        layer.trainable = False
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    return model

model = build_transfer_model(vgg16_ft)


# In[11]:


# モデルのコンパイル
model.compile(
    loss="binary_crossentropy",
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=["accuracy"]
)
model.summary()


# In[12]:


# 学習用画像をロードするためのジェネレータ
idg_train = ImageDataGenerator(
    rescale=1/255.,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)


# In[13]:


# 画像をロードするためのイテレータ
# 訓練用
img_itr_train = idg_train.flow_from_directory(
    "/home/ec2-user/SageMaker/StartTensorFlow/data/input",
    target_size=(224,224),
    batch_size=16,
    class_mode="binary"
)
# 検証用
img_itr_validation = idg_train.flow_from_directory(
    "/home/ec2-user/SageMaker/StartTensorFlow/data/input",
    target_size=(224,224),
    batch_size=16,
    class_mode="binary"
)


# In[14]:


model_dir = os.path.join(
    "../models",
    datetime.now().strftime("%y%m%d_%H%M%S")
)
os.makedirs(model_dir, exist_ok=True)
print("model_dir", model_dir)
dir_weights = os.path.join(model_dir, "weights")
os.makedirs(dir_weights, exist_ok=True)


# In[15]:


# ネットワークの保存
model_json = os.path.join(model_dir, "model_json")
with open(model_json, "w") as f:
    json.dump(model.to_json(), f)
# 学習時の正解ラベルの保存
model_classes = os.path.join(model_dir, "classes.pkl")
with open(model_classes, "wb") as f:
    pickle.dump(img_itr_train.class_indices, f)


# In[16]:


# 1エポックの計算
batch_size = 16
steps_per_epoch = math.ceil(
    img_itr_train.samples/batch_size
)
validation_steps = math.ceil(
    img_itr_validation.samples/batch_size
)


# In[17]:


# Callbacks の設定
cp_filepath = os.path.join(
    dir_weights,
    "ep_{epoch:02d}_ls_{loss:.1f}.h5")
cp = ModelCheckpoint(
    cp_filepath,
    monitor="loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode="auto",
    period=5
)
csv_filepath = os.path.join(model_dir, "loss.csv")
csv = CSVLogger(csv_filepath, append=True)


# In[18]:


# モデルの学習
n_epoch = 30
history = model.fit_generator(
    img_itr_train,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epoch,
    validation_data=img_itr_validation,
    validation_steps=validation_steps,
    callbacks=[cp, csv]
)


# In[19]:


# 予測
test_data_dir = "../data/input"
x_test, true_labels = load_random_imgs(
    test_data_dir,
    seed=1
)
x_test_preproc = preprocess_input(x_test.copy())/255.
probs = model.predict(x_test_preproc)
probs


# In[20]:


# 表示
show_test_samples(
    x_test, probs,
    img_itr_train.class_indices,
    true_labels
)

