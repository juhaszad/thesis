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

# %% [markdown]
# ## Import packages

# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

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
    names = []
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
            names.append(filename)
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
    return (X_train, y_train, names)


# %% [markdown]
# ## Make training set

# %%
X, y, names = make_training_set(filepath)


# %% [markdown]
# ## Check the made training sets

# %%
def properties(array):
    uniq=np.unique(array)
    print("dtype: "+str(array.dtype))
    print("size: "+str(array.size))
    print("shape: "+str(array.shape))
    print("min: "+str(array.min()))
    print("max: "+str(array.max()))
    print("mean: "+str(array.mean()))
    print("std: "+str(array.std()))
    print("len(np.unique): "+str(len(uniq)))
    print("np.unique: ", uniq)
    pass


# %%
properties(X)

# %%
properties(y)


# %% [markdown]
# ### Plotting an example image

# %%
def plot(names, X_train, y_train, i):
    plt.subplot(121)
    plt.title(names[i])
    plt.imshow(np.squeeze(X_train[i], axis=2))
    plt.subplot(122)
    plt.imshow(np.squeeze(y_train[i], axis=2))
    plt.show()
    pass


# %%
plot(names, X, y, 153)

# %% [markdown]
# ## Writing the data into files

# %% [markdown]
# ### Writing the whole dataset into files

# %%
np.save('X_train', X)

# %%
np.save('y_train', y)

# %%
np.save('names', names)

# %% [markdown]
# ### Splitting the dataset into train and test set

# %%
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, names, test_size=0.2)

# %% [markdown]
# #### Standardize

# %%
mean = X_train.mean()
std = X_train.std()

# %%
X_train = (X_train-mean)/std
X_test = (X_test-mean)/std

# %% [markdown]
# ### Fixing the splitted sets for later use

# %%
np.save('fixed_X_train', X_train)
np.save("fixed_y_train", y_train)
np.save("fixed_X_test", X_test)
np.save("fixed_y_test", y_test)
np.save("fixed_train_names", train_index)
np.save("fixed_test_names", test_index)
