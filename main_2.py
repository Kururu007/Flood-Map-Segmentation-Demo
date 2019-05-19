import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('floodmap.jpg')
img_smoothed = cv2.GaussianBlur(img, (7, 7), 0)
img = img_smoothed
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

rows, cols = img.shape
mask = np.zeros((rows, cols, 2), np.uint8)
mask[0, :, :] = 1
mask[:, 0, :] = 1

img_Q = cv2.idft(img_dft * mask)
img_Q = cv2.magnitude(img_Q[:, :, 0], img_Q[:, :, 1])
plt.figure()
fig1 = plt.gcf()
plt.imshow(img_Q)
plt.show()
plt.draw()
fig1.savefig("1.png")

s_x = np.sum(img_Q, 0)
s_y = np.sum(img_Q, 1)
s_x_1 = np.gradient(s_x)
s_x_2 = np.gradient(s_x, 2)
s_y_1 = np.gradient(s_y)
s_y_2 = np.gradient(s_y, 2)
def find_zero_cross(vector):
    res = []
    for i in range(0, len(vector) - 1):
        if vector[i] * vector[i+1] < 0:
            # res.append(i)
            res.append(i+1)
    return res
s_x_1_zero_cross_index = find_zero_cross(s_x_1)
s_y_1_zero_cross_index = find_zero_cross(s_y_1)
def find_index(s_2, subset_index):
    res_index = []
    for s_index in subset_index:
        if s_2[s_index] < 0:
            res_index.append(s_index)
    return res_index
x_index = find_index(s_x_1, s_x_1_zero_cross_index)
y_index = find_index(s_y_1, s_y_1_zero_cross_index)
x_index = np.unique(x_index, axis=0)
y_index = np.unique(y_index, axis=0)
res2 = np.ones((rows, cols))
for x in x_index:
    for y in y_index:
        res2[y, x] = 0

plt.figure()
fig2 = plt.gcf()
plt.imshow(res2, cmap='gray')
plt.show()
plt.draw()
fig2.savefig("2.png")

kernel = np.ones((5,5), np.float32) / 25
res2 = cv2.filter2D(res2, -1, kernel)
res2 = cv2.GaussianBlur(res2, (5, 5), 0)
plt.figure()
fig3 = plt.gcf()
plt.imshow(res2, cmap='gray')
plt.show()
plt.draw()
fig3.savefig("3.png")

import math
def get_line_index(index):
    res = []
    for i in range(0, len(index) - 1):
        instance = math.ceil((index[i+1] + index[i]) / 2)
        res.append(instance)
    return res

line_x_index = get_line_index(x_index)
line_y_index = get_line_index(y_index)
res3 = np.ones((rows, cols))
res3[line_y_index, :] = 0
res3[:, line_x_index] = 0
kernel = np.ones((5, 5), np.float32) / 25
res3 = cv2.filter2D(res3, -1, kernel)
res3 = cv2.GaussianBlur(res3, (5, 5), 0)
plt.figure()
fig4 = plt.gcf()
plt.imshow(res3, cmap='gray')
plt.show()
plt.draw()
fig4.savefig("4.png")

import seaborn as sns
sns.set()
mu, sigma = 30, 7.5
s = np.random.normal(mu, sigma, (450, 450))
s = np.abs(s)
s = np.sort(s)
s = s.transpose()
s[line_y_index, :] = 0
s[:, line_x_index] = 0
ax = sns.heatmap(s, cmap='jet', xticklabels=False, yticklabels=False)
plt.show()
fig = ax.get_figure()
fig.savefig('5.png')
