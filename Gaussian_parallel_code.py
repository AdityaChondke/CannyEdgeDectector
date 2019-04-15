# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 01:22:46 2019

@author: Aditya Chondke
"""

import numpy as np
import cv2
from numba import jit,double
import time


def gaussian(image,fil):
    image_out = np.array(image.copy())

    (h,w) = image.shape
    (hf,wf)=fil.shape
    hf2=hf//2
    wf2=wf//2
    
    for i in range(hf2, h-hf2):
      for j in range(wf2, w-wf2):
        tsum=0
        for ii in range(hf):
            for jj in range(wf):
                tsum=tsum+(image[i-hf2+ii,j-wf2+jj]*fil[hf-1-ii,wf-1-jj])
                
        image_out[i][j]=tsum
    
    return image_out



image =np.array(cv2.imread('Input Image path',cv2.IMREAD_GRAYSCALE))
gauss2=np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])

gaussian_fast = jit(double[:,:](double[:,:], double[:,:]))(gaussian)

start=time.time()
image_out=gaussian_fast(image,gauss2)
end=time.time()          
   
print(end-start) 
cv2.imwrite('Output image path' , image_out)


