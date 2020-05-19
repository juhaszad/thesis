# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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
run_name = 'original_masks_model'
log_dir = os.path.normpath(os.path.join('./Tensorboard/',run_name))

# %%
filepath="./"

# %%
X = np.load(filepath + "X_train.npy")
y = np.load(filepath + "y_train.npy")
names = np.load(filepath + "names.npy")

# %% [markdown]
# Use original masks

# %%
X = np.load(filepath + "X_train.npy")
y = np.load(filepath + "y_train_original.npy")
names = np.load(filepath + "names.npy")

# %%
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, names, test_size=0.2)

# %% [markdown]
# Standardization

# %%
mean = X_train.mean()
std = X_train.std()

# %%
X_train = (X_train-mean)/std
X_test = (X_test-mean)/std

# %% [markdown]
# Normalization

# %%
maximum = X_train.max()
minimum = X_train.min()

# %%
X_train = (X_train-minimum)/(maximum-minimum)
X_test = (X_test-minimum)/(maximum-minimum)

# %%
np.save('fixed_X_train', X_train)
np.save("fixed_y_train", y_train)
np.save("fixed_X_test", X_test)
np.save("fixed_y_test", y_test)
np.save("fixed_train_names", train_index)
np.save("fixed_test_names", test_index)

# %%
X_train = np.load('fixed_X_train.npy')
y_train = np.load('fixed_y_train.npy')
X_test = np.load('fixed_X_test.npy')
y_test = np.load('fixed_y_test.npy')
train_index = np.load('fixed_train_names.npy')
test_index = np.load('fixed_test_names.npy')

# %%
plt.hist(np.log(X_train.ravel()+1))

# %%
log_X_train = np.log(X_train+1)
log_X_test = np.log(X_test+1)

# %%
plt.subplot(121)
plt.imshow(np.squeeze(log_X_train[62], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[62], axis=2))

# %%
plt.subplot(121)
plt.imshow(np.squeeze(X_train[62], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[62], axis=2))
plt.show()
plt.subplot(121)
plt.imshow(np.squeeze(X_train[0], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[0], axis=2))
plt.show()
plt.subplot(121)
plt.imshow(np.squeeze(X_train[700], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[700], axis=2))
plt.show()

# %%
del X, y


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
def dice(y_true, y_pred):
    TP = keras.backend.sum(y_true*y_pred)
    FP_FN = keras.backend.sum(keras.backend.abs(y_true-y_pred))
    return 2*TP/(2*TP+FP_FN)


# %%
def dice_loss(y_true,y_pred):
    return 1-dice(y_true,y_pred)


# %%
class binary_crossentropy:
    def __init__(self, w):
        self.w=w
        self.name="custom_loss"
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred=tf.clip_by_value(y_pred, keras.backend.epsilon(), 1-keras.backend.epsilon())
        loss=tf.math.reduce_mean((-1)*(self.w*y_true*tf.math.log(y_pred)+(1-y_true)*tf.math.log(1-y_pred)))
        return loss


# %%
class new_custom_loss:
    def __init__(self, w):
        self.w=w
        self.name="new_custom_loss"
    def __call__(self, y_true, y_pred, sample_weight=None):
        FP = tf.math.count_nonzero(y_pred * (y_true - 1))
        FN = tf.math.count_nonzero((y_pred - 1) * y_true)
        size = tf.size(y_true, out_type=tf.dtypes.int64, name=None)
        return tf.math.divide(keras.backend.sum(FP, tf.math.scalar_mul(self.w, FN)),size)


# %%
custom_loss = binary_crossentropy(10)

# %%
cl = new_custom_loss(10)

# %%
del model

# %%
model = unet_model()

# %%
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')
lr = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0)

# %%
model.compile(optimizer = keras.optimizers.Adam(lr = 1e-2), loss = custom_loss, metrics = ['accuracy', dice])

# %%
model.fit(X_train, y_train, epochs = 50, batch_size=1, callbacks=[tensorboard_callback, lr])

# %%
model2 = keras.models.load_model('./Model_lr_bigger.h5', custom_objects={'dice': dice})

# %%
del X_train, y_train, y_test

# %%
prediction = model.predict(X_test, batch_size=1)

# %%
np.save('Predictions', prediction)
np.save('test_names', test_index)

# %%
plt.subplot(121)
plt.title(str(test_index[93]))
plt.imshow(np.squeeze(X_test[93], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(prediction[93], axis=2))
plt.show()
plt.subplot(121)
plt.title(str(test_index[0]))
plt.imshow(np.squeeze(X_test[0], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(prediction[0], axis=2))
plt.show()
plt.subplot(121)
plt.title(str(test_index[103]))
plt.imshow(np.squeeze(X_test[103], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(prediction[103], axis=2))
plt.show()
plt.subplot(121)
plt.title(str(test_index[62]))
plt.imshow(np.squeeze(X_test[62], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(prediction[62], axis=2))
plt.show()

# %%
prediction[prediction>=0.5]=1
prediction[prediction<0.5]=0

# %%
cm=confusion_matrix(y_test.ravel(), prediction.ravel())

# %%
cm

# %%
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# %%
annot_kws = {"ha": 'left',"va": 'top'}
ax = sns.heatmap(cmn, annot=True, annot_kws=annot_kws, cmap=plt.cm.Blues, fmt='.4f')

# %%
fpr, tpr, thresholds = roc_curve(y_test.ravel(), prediction.ravel())

# %%
plt.plot(fpr,tpr)

# %%
model.evaluate(X_test, y_test, batch_size=1)

# %%
model.save('Model_fixed_log.h5')

# %%
model.save_weights('Weights_fixed_log.h5')


# %%
def TP(y_true, y_pred):
    return np.sum(np.around(y_pred)* y_true)


def TN(y_true, y_pred):
    return np.sum((np.around(y_pred) + y_true ) == 0)


def FP(y_true, y_pred):
    return np.sum(np.around(y_pred) - y_true)


def FN(y_true, y_pred):
    return np.sum((y_true - np.around(y_pred)) == 1)


def Specificity(y_true, y_pred):
    return TN(y_true, y_pred) / (np.sum(1 - np.around(y_pred)) + keras.backend.epsilon())


def Sensitivity(y_true, y_pred):
    return TP(y_true, y_pred) / (np.sum(np.around(y_pred)) + keras.backend.epsilon())

# %%
