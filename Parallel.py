# Serial Gaussian Filter

import numpy as np
import cv2

image=cv2.imread('file path here'),cv2.IMREAD_GRAYSCALE)
image_out = image.copy()

h = image.shape(0)
w = image.shape(1)

gauss = (1.0/57)*np.array([[0, 1, 2, 1, 0], [1, 3, 5, 3, 1], [2, 5, 9, 5, 2], [1, 3, 5, 3, 1], [0, 1, 2, 1, 0]])
sum(sum(gauss))

for i in np.arange(2, height-2):
  for j in np.arange(2, width-2):
    sum = 0
    for k in np.arange(-2, 3):
      for l in np.arange(-2, 3):
        a = image.item(i+k, j+l)
        p = gauss[2+k, 2+l]
        sum - sum + (p * a)
    b = sum
    image_out.itemset((i, j), b)
    
cv2.imwrite('file path here' , image_out)

#cv2.imshow('image', image_out)
#cv2.waitKey(0)





