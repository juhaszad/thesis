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

# %%
predictions = np.load('./Predictions.npy')

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

# %%
_, e, _ = cv.findContours(canvas, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)

# %%
canvas_e = np.zeros_like(canvas)
cv.drawContours(canvas_e , e[0], -1, (255, 255, 255), 3)

plt.imshow(canvas_e)

# %%
canvas_e = np.zeros_like(canvas)
cv.drawContours(canvas_e , e[1], -1, (255, 255, 255), 3)

plt.imshow(canvas_e)

# %%
cv.arcLength(e[0], True)

# %%
cv.arcLength(e[1], True)

# %%
