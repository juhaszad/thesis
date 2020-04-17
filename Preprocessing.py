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
# ## Experimenting, getting to know the data

# %%
filepath="./Data/training_set/"

# %%
filename="010_HC.png"

# %%
img=cv.imread(filepath+filename, 0)

# %%
new_img=np.expand_dims(img, axis=2)

# %%
new_img.shape

# %%
res=cv.resize(new_img,(640,432))

# %%
res=np.expand_dims(res, axis=2)

# %%
res.shape

# %%
plt.imshow(img)

# %%
img_normed=(img-img.mean())/img.std()

# %%
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.hist(img.ravel(), bins=256, range=(-20.0, 250.0))
plt.title("Eredeti")
plt.subplot(122)
plt.hist(img_normed.ravel(), bins=256, range=(-10.0, 10.0))
plt.title("Normalizált")
plt.show()

# %%
img_normed.shape

# %%
resized=cv.resize(img_normed, (640,432))

# %%
resized.shape

# %%
plt.imshow(resized)

# %%
edges=cv.Canny(img, 20, 20)

# %%
plt.imshow(edges)

# %% [markdown]
# ## Reading all data

# %%
filepath="./Data/training_set/"

# %%
filenames= os.listdir(filepath)
filenames.sort()
filenames

# %%
elso=os.listdir(filepath)[0]
elso[-14:]

# %%
masodik=os.listdir(filepath)[2]
masodik[-6:]

# %%
np.zeros((999,432,640,1))[0].shape


# %%
def make_training_set(filepath):
    filenames = os.listdir(filepath)
    filenames.sort()
    X_train = np.zeros((999,432,640,1))
    y_train = np.zeros((999,432,640,1))
    index_X = 0
    index_y = 0
    for filename in filenames:
        if filename[-6:]=='HC.png':
            X = cv.imread(filepath+filename, 0)
            X = (X-X.mean())/X.std()
            X = cv.resize(X, (640,432))
            X = np.expand_dims(X, axis=2)
            X_train[index_X] = X
            index_X+=1
        elif filename[-14:]=='Annotation.png':
            y = cv.imread(filepath+filename, 0)
            y = cv.resize(y, (640,432))
            y = np.expand_dims(y, axis=2)
            y_train[index_y] = y
            index_y+=1
    return (X_train, y_train)


# %%
X_train, y_train = make_training_set(filepath)

# %%
len(X_train)

# %%
X_train[0].shape

# %%
len(y_train)

# %%
plt.subplot(121)
plt.imshow(np.squeeze(X_train[153], axis=2))
plt.subplot(122)
plt.imshow(np.squeeze(y_train[153], axis=2))
plt.show()

# %% [markdown]
# ## Writing the data into files

# %%
np.save('X_train', X_train)

# %%
np.save('y_train', y_train)
