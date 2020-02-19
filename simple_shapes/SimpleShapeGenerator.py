import numpy as np
from skimage import draw as dr
import math
import matplotlib.pyplot as plt

data = np.zeros((200,200,4), dtype=np.uint8)

rr, cc = dr.circle(100, 100, radius = 80, shape = data.shape)
data[rr,cc] = 1
plt.imshow(data)
plt.show()
plt.imsave('test.png', data)
