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
# # Ellipse fit

# %% [markdown]
# ## Import packages

# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special

# %% [markdown]
# ## Load the data

# %%
predictions = np.load('./Prediction_with_bce_2params.npy')

# %%
pred_names = np.load('./fixed_test_names.npy')

# %%
properties = pd.read_csv('./training_set_pixel_size_and_HC.csv')


# %% [markdown]
# ## Define functions for the calculations

# %%
def calc_HC_with_integral(a,b,pixel_size):
    a=a*1.25*pixel_size*0.5
    b=b*1.25*pixel_size*0.5
    e_sq = 1.0-b**2/a**2
    HC = 4*a*special.ellipe(e_sq)
    return HC


# %%
def get_pixel_size(filename):
    return float(properties[properties['filename']==str(filename)]['pixel size(mm)'])


# %% [markdown]
# ## Ellipse fitting and calculation of the HC

# %%
def EllipseFit(predictions, pred_names):
    d = dict()
    for i, pred in enumerate(predictions):
        trial = np.squeeze(pred.copy(), axis=2)
        trial[trial<0.5]=0
        trial[trial>=0.5]=1
        trial = np.uint8(trial)
        img, contours, _ = cv.findContours(trial, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)
        conts = filter(lambda x: len(x) >= 5, contours)
        ellipses = list(map(cv.fitEllipse, conts))
        a = 0
        b = 0
        for e in ellipses:
            if e[1][0]>a:
                a=e[1][0]
            if e[1][1]>b:
                b=e[1][1]
        pixel_size = get_pixel_size(pred_names[i])
        HC = calc_HC_with_integral(a,b,pixel_size)
        d[pred_names[i]]=HC
    return d


# %%
d = EllipseFit(predictions, pred_names)


# %% [markdown]
# ## Calculating the difference

# %%
def pd_to_dict(df):
    d = dict()
    l = df.values.tolist()
    for i in range(len(l)):
        d[l[i][0]]=l[i][2]
    return d


# %%
prop_dict = pd_to_dict(properties)


# %%
def mean_absolute_difference(d, prop_dict):
    difference = 0
    for filename in list(d.keys()):
        difference += abs(d[filename]-prop_dict[filename])
    return difference/len(d)


# %%
MADF = mean_absolute_difference(d, prop_dict)

# %%
print("The mean absolute difference measured on the test set is:" + str(MADF))
