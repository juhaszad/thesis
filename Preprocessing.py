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
# # Data preprocessing

# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# %% [markdown]
# ## Reading all data

# %%
filepath="./Data/"

# %%
height=432
width=640
set_size=int(len(os.listdir(filepath))/2)

# %%
filenames= os.listdir(filepath)
filenames.sort()
filenames


# %%
def make_training_set(filepath):
    filenames = os.listdir(filepath)
    filenames.sort()
    X_train = np.zeros((set_size,height,width,1), dtype='float32')
    y_train = np.zeros((set_size,height,width,1), dtype='float32')
    index_X = 0
    index_y = 0
    for filename in filenames:
        if filename[-6:]=='HC.png':
            X = cv.imread(filepath+filename, 0)
            X = cv.resize(X, (width,height))
            X = np.expand_dims(X, axis=2)
            X_train[index_X] = X
            index_X+=1
        elif filename[-14:]=='Annotation.png':
            y = cv.imread(filepath+filename, 0)
            y = y/255
            y = cv.resize(y, (width,height), interpolation=cv.INTER_NEAREST)
            y = np.expand_dims(y, axis=2)
            mask = np.zeros((height+2, width+2), np.uint8)
            cv.floodFill(np.uint8(y), mask, (0,0), 1)
            inverted = cv.bitwise_not(mask)
            inverted = inverted-254
            im_out = inverted[1:height+1,1:width+1]
            y=np.float32(im_out)
            y = np.expand_dims(y, axis=2)
            y_train[index_y] = y
            index_y+=1
    return (X_train, y_train)


# %% [markdown]
# Make training set

# %%
X_train, y_train = make_training_set(filepath)

# %% [markdown]
# Examine made training sets

# %%
uniq_X=np.unique(X_train)

# %%
uniq_y=np.unique(y_train)

# %%
plt.subplot(121)
plt.imshow(np.squeeze(X_train[153], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[153], axis=2))
plt.show()

# %%
print("X_train")
print("dtype: "+str(X_train.dtype))
print("size: "+str(X_train.size))
print("shape: "+str(X_train.shape))
print("min: "+str(X_train.min()))
print("max: "+str(X_train.max()))
print("mean: "+str(X_train.mean()))
print("std: "+str(X_train.std()))
print("len(np.unique): "+str(len(uniq_X)))
print("np.unique: ", uniq_X)

# %%
print("y_train")
print("dtype: "+str(y_train.dtype))
print("size: "+str(y_train.size))
print("shape: "+str(y_train.shape))
print("min: "+str(y_train.min()))
print("max: "+str(y_train.max()))
print("mean: "+str(y_train.mean()))
print("std: "+str(y_train.std()))
print("len(np.unique): "+str(len(uniq_y)))
print("np.unique: "+str(uniq_y))

# %% [markdown]
# ## Writing the data into files

# %%
np.save('X_train', X_train)

# %%
np.save('y_train', y_train)

# %%
