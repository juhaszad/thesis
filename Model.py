# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model building and training

# %% [markdown]
# ## Import packages

# %%
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

# %% [markdown]
# ## Set TensorBoard directory

# %%
run_name = 'bce_two_params'
log_dir = os.path.normpath(os.path.join('./Tensorboard/',run_name))

# %% [markdown]
# ## Load the (fixed) data

# %%
X_train = np.load('fixed_X_train.npy')
y_train = np.load('fixed_y_train.npy')
X_test = np.load('fixed_X_test.npy')
y_test = np.load('fixed_y_test.npy')
train_index = np.load('fixed_train_names.npy')
test_index = np.load('fixed_test_names.npy')


# %% [markdown]
# ### Plotting some images

# %%
def plot(names, X_train, y_train, i):
    plt.subplot(121)
    plt.title(names[i])
    plt.imshow(np.squeeze(X_train[i], axis=2))
    plt.subplot(122)
    plt.imshow(np.squeeze(y_train[i], axis=2))
    plt.show()


# %%
plot(names, X_train, y_train, 62)


# %% [markdown]
# ## Building the model

# %%
def unet_model(input_size = (432,640,1)):
    inputs = keras.layers.Input(input_size)
    
    # Contraction path
    
    conv11 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(inputs)
    conv12 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same')(conv11)
    
    pool1 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv12)
    
    conv21 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(pool1)
    conv22 =keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv21)
    
    pool2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv22)
    
    conv31 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(pool2)
    conv32 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv31)
    
    pool3 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv32)
    
    conv41 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(pool3)
    conv42 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv41)
    
    pool4 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(conv42)
    
    conv51 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(pool4)
    conv52 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv51)
    
    #Expansion path
    
    up6 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(keras.layers.UpSampling2D(size=(2,2))(conv52))
    merge6 = keras.layers.concatenate([conv42, up6], axis = 3)
    
    conv61 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(merge6)
    conv62 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv61)
    
    up7 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(keras.layers.UpSampling2D(size = (2,2))(conv62))
    merge7 = keras.layers.concatenate([conv32, up7], axis = 3)
    
    conv71 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(merge7)
    conv72 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv71)
    
    up8 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(keras.layers.UpSampling2D(size = (2,2))(conv72))
    merge8 = keras.layers.concatenate([conv22, up8], axis = 3)
    
    conv81 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(merge8)
    conv82 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv81)
    
    up9 = keras.layers.Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(keras.layers.UpSampling2D(size=(2,2))(conv82))
    merge9 = keras.layers.concatenate([conv12, up9], axis = 3)
    
    conv91 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(merge9)
    conv92 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv91)
    conv93 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer=keras.initializers.he_normal())(conv92)
    
    conv10 = keras.layers.Conv2D(1,1, activation = 'sigmoid', kernel_initializer=keras.initializers.he_normal())(conv93)
    
    model = keras.models.Model(inputs = inputs, outputs = conv10)
    
    return model


# %% [markdown]
# ### Introducing a new metric

# %%
def dice(y_true, y_pred):
    TP = keras.backend.sum(y_true*y_pred)
    FP_FN = keras.backend.sum(keras.backend.abs(y_true-y_pred))
    return 2.0*TP/(2.0*TP+FP_FN)


# %% [markdown]
# ### Introducing new loss functions

# %%
def dice_loss(y_true,y_pred):
    return 1.0-dice(y_true,y_pred)


# %%
class binary_crossentropy:
    def __init__(self, w_FP, w_FN):
        self.w_FP=w_FP
        self.w_FN=w_FN
        self.__name__="custom_loss"
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred=tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0-keras.backend.epsilon())
        loss=tf.math.reduce_mean((self.w_FN*y_true*tf.math.log(y_pred)+self.w_FP*(1.0-y_true)*tf.math.log(1.0-y_pred)))
        return loss*(-1.0)


# %%
class weighted_fn_fp:
    def __init__(self, w_FP, w_FN):
        self.w_FP=w_FP
        self.w_FN=w_FN
        self.__name__="weighted_fn_fp"
    def __call__(self, y_true, y_pred, sample_weight=None):
        FP = (y_pred * (1.0-y_true))
        FN = ((1.0-y_pred) * y_true)
        size = tf.dtypes.cast(tf.reduce_prod(tf.shape(y_pred)), tf.dtypes.float32)
        return (self.w_FP*keras.backend.sum(FP)+self.w_FN*keras.backend.sum(FN))/size


# %% [markdown]
# ## Training the model

# %%
model = unet_model()

# %%
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

# %%
model.compile(optimizer = keras.optimizers.Adam(), loss = weighted_fn_fp(w_FP=0.3, w_FN=0.7), metrics = ['accuracy', dice])

# %%
model.fit(X_train, y_train, epochs = 50, batch_size=2, callbacks=[tensorboard_callback])

# %% [markdown]
# ## Making predictions

# %%
prediction = model.predict(X_test, batch_size=1)

# %% [markdown]
# #### Saving the prediction for later use

# %%
np.save('Prediction_with_bce_2params', prediction)

# %% [markdown]
# #### Plotting some of the predictions

# %%
plot(test_index, X_test, prediction, 130)
plot(test_index, X_test, prediction, 199)

# %% [markdown]
# ## Evaluating the prediction

# %% [markdown]
# ### Standard evaluation

# %%
model.evaluate(X_test, y_test, batch_size=1)

# %% [markdown]
# ### Visual evaluation

# %% [markdown]
# #### Confusion matrix

# %%
prediction[prediction>=0.5]=1
prediction[prediction<0.5]=0

# %%
cm=confusion_matrix(y_test.ravel(), prediction.ravel())

# %%
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# %%
annot_kws = {"ha": 'left',"va": 'top'}
ax = sns.heatmap(cmn, annot=True, annot_kws=annot_kws, cmap=plt.cm.Blues, fmt='.4f')
ax.set_title("Confusion matrix")
plt.show()

# %% [markdown]
# #### ROC curve

# %%
fpr, tpr, thresholds = roc_curve(y_test.ravel(), prediction.ravel())


# %%
def plot_roc_auc(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of the reduced model')
    plt.legend(loc="lower right")
    plt.show()


# %%
plot_roc_auc(fpr, tpr)

# %% [markdown]
# ## Saving the model and the weights

# %%
model.save('Model_bce_2params.h5')

# %%
model.save_weights('Weights_bce_2params.h5')
