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

# %% [markdown]
# ## Experimenting, getting to know the data

# %%
filepath="./Data/training_set/"

# %%
filename="010_HC.png"

# %%
img=cv.imread(filepath+filename, 0)

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
import os

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
def make_training_set(filepath):
    filenames = os.listdir(filepath)
    filenames.sort()
    X_train = []
    y_train = []
    for filename in filenames:
        if filename[-6:]=='HC.png':
            X = cv.imread(filepath+filename, 0)
            X = (X-X.mean())/X.std()
            X = cv.resize(X, (640,432))
            X_train.append(X)
        elif filename[-14:]=='Annotation.png':
            y = cv.imread(filepath+filename, 0)
            y = (y-y.mean())/y.std()
            y = cv.resize(y, (640,432))
            y_train.append(y)
    return (X_train, y_train)


# %%
X_train, y_train = make_training_set(filepath)

# %%
len(X_train)

# %%
len(y_train)

# %%
plt.subplot(121)
plt.imshow(X_train[152])
plt.subplot(122)
plt.imshow(y_train[152])
plt.show()

# %% [markdown]
# ## Writing the data into files

# %%
import pickle

# %%
with open('X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)
f.close()

# %%
with open('y_train.pickle', 'wb') as g:
    pickle.dump(y_train, g)
g.close()
