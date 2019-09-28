#%%

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

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
anchor_scales = [2, 4, 8]
receptive_size = 16

ctx_x = np.arange(0, img_data.shape[1], receptive_size)
ctx_y = np.arange(0, img_data.shape[0], receptive_size)

ctx_x, ctx_y = np.meshgrid(ctx_x, ctx_y)
ctx = np.stack((ctx_x, ctx_y), axis=2).reshape(-1, 2)
ctx.shape

#%%
archor_idx = random.randint(0, ctx.shape[0])
archor_point = ctx[archor_idx, :]
# x, y, w, h
archor_box = np.zeros([9, 4])
rate, scales = np.meshgrid(ratios, anchor_scales)
ras = np.stack((rate, scales), axis=2).reshape(-1, 2)
for idx in range(ras.shape[0]):
    archor_box_width = ras[idx, 1] * receptive_size / np.sqrt(ras[idx, 0])
    archor_box_high = ras[idx, 1] * receptive_size * np.sqrt(ras[idx, 0])
    archor_box[idx, 0] = archor_point[0] - archor_box_width / 2
    archor_box[idx, 1] = archor_point[1] - archor_box_high / 2
    archor_box[idx, 2] = archor_box_width
    archor_box[idx, 3] = archor_box_high

img_show = img.copy()
draw = ImageDraw.Draw(img_show)
for idx in range(archor_box.shape[0]):
    draw.rectangle([(archor_box[idx, 0], archor_box[idx, 1]), (archor_box[idx, 0] + archor_box[idx, 2], archor_box[idx, 1] + archor_box[idx, 3])], outline=(0, 255, 0))
plt.figure(figsize=(15, 15))
plt.imshow(img_show)

#%%
archor_box = np.zeros([ctx.shape[0]*9, 4])
for ctx_id in range(ctx.shape[0]):
    for idx in range(ras.shape[0]):
        archor_box_width = ras[idx, 1] * receptive_size / np.sqrt(ras[idx, 0])
        archor_box_high = ras[idx, 1] * receptive_size * np.sqrt(ras[idx, 0])
        archor_box[ctx_id * 9 + idx, 0] = archor_point[0] - archor_box_width / 2
        archor_box[ctx_id * 9 + idx, 1] = archor_point[1] - archor_box_high / 2
        archor_box[ctx_id * 9 + idx, 2] = archor_box_width
        archor_box[ctx_id * 9 + idx, 3] = archor_box_high

img_show = img.copy()
draw = ImageDraw.Draw(img_show)
for idx in range(archor_box.shape[0]):
    draw.rectangle([(archor_box[idx, 0], archor_box[idx, 1]), (archor_box[idx, 0] + archor_box[idx, 2], archor_box[idx, 1] + archor_box[idx, 3])], outline=(0, 255, 0))
plt.figure(figsize=(15, 15))
plt.imshow(img_show)



#%%
