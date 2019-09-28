#%%

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#%%
img = Image.open("./dataset/test.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(img)


#%%
targets = {"human":[[229, 68, 250, 434]], "dog":[[60, 325, 124, 180]]}
img_show = img.copy()
draw = ImageDraw.Draw(img_show)
for key in targets:
    for rec in targets[key]:
        draw.rectangle([(rec[0], rec[1]), (rec[0]+rec[2], rec[1]+rec[3])], outline=(255, 0, 0))
plt.figure(figsize=(15, 15))
plt.imshow(img_show)
#%%
img_data = np.array(img)
ratios = [0.5, 1, 2]
anchor_scales = [4, 8, 16]
receptive_size = 16

ctx_x = np.arange(0, img_data.shape[1], receptive_size)
ctx_y = np.arange(0, img_data.shape[0], receptive_size)

ctx_x, ctx_y = np.meshgrid(ctx_x, ctx_y)
ctx = np.stack((ctx_x, ctx_y), axis=2).reshape(-1, 2)
ctx.shape

#%%
