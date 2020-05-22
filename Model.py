# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model

# %%
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

# %%
run_name = 'a'
log_dir = os.path.normpath(os.path.join('/net/home/david/adel/tensorboard/',run_name))

# %%
filepath="/net/home/david/adel/"

# %%
X = np.load(filepath + "X_train.npy")
y = np.load(filepath + "y_train.npy")

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
plt.subplot(121)
plt.imshow(np.squeeze(X_train[62], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[62], axis=2))
plt.show()


# %%
def unet_model(input_size = (432,640,1)):
    inputs = keras.layers.Input(input_size)
    
    # Contraction path
    
    conv11 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv12 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(conv11)
    
    pool1 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv12)
    
    conv21 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv22 =keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv21)
    
    pool2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv22)
    
    conv31 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv32 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv31)
    
    pool3 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv32)
    
    conv41 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv42 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(conv41)
    
    pool4 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv42)
    
    conv51 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv52 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv51)
    
    #Expansion path
    
    up6 = keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same')(keras.layers.UpSampling2D(size=(2,2))(conv52))
    merge6 = keras.layers.concatenate([conv42, up6], axis = 3)
    
    conv61 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv62 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(conv61)
    
    up7 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same')(keras.layers.UpSampling2D(size = (2,2))(conv62))
    merge7 = keras.layers.concatenate([conv32, up7], axis = 3)
    
    conv71 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv72 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv71)
    
    up8 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same')(keras.layers.UpSampling2D(size = (2,2))(conv72))
    merge8 = keras.layers.concatenate([conv22, up8], axis = 3)
    
    conv81 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv82 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv81)
    
    up9 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same')(keras.layers.UpSampling2D(size=(2,2))(conv82))
    merge9 = keras.layers.concatenate([conv12, up9], axis = 3)
    
    conv91 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv92 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(conv91)
    conv93 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same')(conv92)
    
    conv10 = keras.layers.Conv2D(1,1, activation = 'sigmoid')(conv93)
    
    model = keras.models.Model(inputs = inputs, outputs = conv10)
    
    return model


# %%
class binary_crossentropy:
    def __init__(self, w):
        self.w=w
        self.__name__="custom_loss"
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred=tf.clip_by_value(y_pred, keras.backend.epsilon(), 1-keras.backend.epsilon())
        loss=tf.math.reduce_mean((-1)*(self.w*y_true*tf.math.log(y_pred)+(1-y_true)*tf.math.log(1-y_pred)))
        return loss


# %%
print(y_train.sum() / y_train.size)

# %%
model = unet_model()

# %%
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

# %%
model.compile(optimizer = keras.optimizers.Adam(), loss = binary_crossentropy(3.3/3.0), metrics = ['accuracy'])

# %%
model.fit(X_train, y_train, epochs = 50, batch_size=8,callbacks=[tensorboard_callback])

# %%
prediction = model.predict(X_test[133:134, :, :, 0], batch_size=1)

# %%
plt.subplot(121)
plt.imshow(X_test[133, :, :, 0])
plt.subplot(122)
plt.imshow(prediction[0, :, :, 0])
plt.show()

# %%
