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
# # Ellipse fit

# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

# %%
predictions = np.load('./Predictions.npy')
pred_names = np.load('./test_names.npy')

# %%
properties = pd.read_csv('./training_set_pixel_size_and_HC.csv')

# %%
properties.head()

# %%
float(properties[properties['filename']=='000_HC.png']['pixel size(mm)'])

# %%
trial = predictions[0].copy()

# %%
plt.imshow(np.squeeze(trial, axis=2))

# %%
trial = np.squeeze(trial, axis=2)

# %%
trial[trial<0.5]=0
trial[trial>=0.5]=1
trial = np.uint8(trial)

# %%
img, contours, _ = cv.findContours(trial, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)

# %%
plt.imshow(img)

# %%
canvas = np.zeros_like(img)
cv.drawContours(canvas , contours, -1, (255, 255, 255), 3)

plt.imshow(canvas)

# %%
for i, contour in enumerate(contours):
    e = cv.fitEllipse(contour)

# %%
conts = filter(lambda x: len(x) >= 40, contours)
ellipses = list(map(cv.fitEllipse, conts)) 

# %%
ellipses

# %%
canvas = np.zeros_like(img)
cv.ellipse(canvas, ellipses[0], (255, 0, 0), 3)
plt.imshow(canvas)


# %% [markdown]
# # Ellipse fit, calculations

# %%
def calc_HC(a,b,pixel_size):
    a=a*1.25*pixel_size*0.5
    b=b*1.25*pixel_size*0.5
    HC = np.pi*(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
    return HC


# %%
def get_pixel_size(filename):
    return float(properties[properties['filename']==str(filename)]['pixel size(mm)'])


# %%
def EllipseFit():
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
        HC = calc_HC(a,b,pixel_size)
        d[pred_names[i]]=HC
    return d


# %%
d = EllipseFit()

# %% [markdown]
# # Calculating the difference

# %%
d

# %%
float(properties[properties['filename']=='494_HC.png']['head circumference (mm)'])


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
def mean_absolute_difference():
    difference = 0
    for filename in list(d.keys()):
        difference += abs(d[filename]-prop_dict[filename])
    return difference/len(d)


# %%
MADF = mean_absolute_difference()

# %%
MADF

# %%
