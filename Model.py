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
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras

# %%
filepath="./"

# %%
X = np.load(filepath + "X_train.npy")
y = np.load(filepath + "y_train.npy")

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
mean = X_train.mean()
std = X_train.std()

# %%
X_train = (X_train-mean)/std
X_test = (X_test-mean)/std

# %%
uniq_X = np.unique(X_train)

# %%
print("len(np.unique): "+str(len(uniq_X)))
print("np.unique: ", uniq_X)

# %%
del X, y


# %%
def unet_model(input_size = (216,320,1)):
    inputs = Input(input_size)
    
    # Contraction path
    
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv12 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv11)
    
    pool1 = MaxPooling2D(pool_size=(2,2), strides = 2)(conv12)
    
    conv21 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv22 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv21)
    
    pool2 = MaxPooling2D(pool_size=(2,2), strides = 2)(conv22)
    
    conv31 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv32 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv31)
    
    pool3 = MaxPooling2D(pool_size=(2,2), strides = 2)(conv32)
    
    conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv41)
    
    pool4 = MaxPooling2D(pool_size=(2,2), strides = 2)(conv42)
    
    conv51 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv52 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv51)
    
    #Expansion path
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size=(2,2))(conv52))
    merge6 = concatenate([conv42, up6], axis = 3)
    
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv61)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv62))
    merge7 = concatenate([conv32, up7], axis = 3)
    
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv71)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv72))
    merge8 = concatenate([conv22, up8], axis = 3)
    
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv81)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size=(2,2))(conv82))
    merge9 = concatenate([conv12, up9], axis = 3)
    
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv91)
    conv93 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv92)
    
    conv10 = Conv2D(1,1, activation = 'sigmoid')(conv93)
    
    model = Model(input = inputs, output = conv10)
    
    #model.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


# %%
model = unet_model()

# %%
model.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['accuracy'])

# %%
model.fit(X_train, y_train, epochs = 1)

# %%
model.evaluate(X_test, y_test)

# %%
