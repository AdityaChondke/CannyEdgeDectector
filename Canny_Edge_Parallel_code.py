# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 01:22:46 2019

@author: Aditya Chondke
"""

import numpy as np
import cv2
from numba import jit,double
import time
import math



def gaussian(image):
    fil=np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    image_out = np.array(image.copy())
    (h,w) = image.shape
    (hf,wf)=fil.shape
   
    hf2=hf//2
    wf2=wf//2
    
    for i in range(hf2, h-hf2):
      for j in range(wf2, w-wf2):
        tsum=0
        tsumx=0
        tsumy=0
        for ii in range(hf):
            for jj in range(wf):
                tsum=tsum+(image[i-hf2+ii,j-wf2+jj]*fil[hf-1-ii,wf-1-jj])
        
        image_out[i][j]=tsum
        
    return image_out







def gradient(image):
    image_out2 = np.array(image.copy())
    image_out3 = np.array(image.copy())
    
    gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy=gx.T

    (h,w) = image.shape
    (hf,wf)=gx.shape
   
    hf2=hf//2
    wf2=wf//2
    
    for i in range(hf2, h-hf2):
      for j in range(wf2, w-wf2):
        tsumx=0
        tsumy=0
        for ii in range(hf):
            for jj in range(wf):
                tsumx=tsumx+(image[i-hf2+ii,j-wf2+jj]*gx[hf-1-ii,wf-1-jj])
                tsumy=tsumy+(image[i-hf2+ii,j-wf2+jj]*gy[hf-1-ii,wf-1-jj])
        
                
        image_out2[i][j]=math.sqrt((tsumx*tsumx)+(tsumy*tsumy))
        theta = np.arctan2(tsumy, tsumx)
        image_out3[i][j] = (np.round(theta * (5.0 / np.pi)) + 5) % 5   #angle quantization 
   
    return (image_out2,image_out3)



def nonmaxima(image,imageQ):
    image_out_nmax = np.array(image.copy())
    imagea = np.array(imageQ.copy())
    
    (h,w)=image.shape
    
    for i in range(h):
        for j in range(w):
            if(i==0 or i==h-1 or j==0 or j==w-1 ):
                image_out_nmax[i][j]=0
                continue
            
            tq=(imagea[i][j])%4
            if(tq==0):
                if(image[i,j]<=image[i,j-1] or image[i,j]<=image[i, j+1]):
                    image_out_nmax[i][j]=0
            if(tq==1):
                if(image[i,j]<=image[i-1,j+1] or image[i,j]<=image[i+1,j-1]):
                    image_out_nmax[i][j]=0
            if(tq==2):
                if(image[i,j]<=image[i-1,j] or image[i,j]<=image[i+1,j]):
                    image_out_nmax[i][j]=0
            if(tq == 3):
                if(image[i, j] <= image[i-1, j-1] or image[i, j] <= image[i+1, j+1]):
                    image_out_nmax[i][j]=0
    
            
    return image_out_nmax
    



def hysteresis(imageS,imageT):
    imagefinal = np.array(imageS.copy())
    imageth=np.array(imageT.copy())
    currentp=np.array([(-1,-1)])
    (h,w) = image.shape
    (hf,wf)=(3,3)
   
    hf2=hf//2
    wf2=wf//2
    
    for i in range(hf2, h-hf2):
      for j in range(wf2, w-wf2):
          if(imageth[i][j]!=1):
              continue
          for ii in range(hf):
              for jj in range(wf):
                  if(imageth[i-hf2+ii,j-wf2+jj]==2):
                      currentp=np.append(currentp,[(i,j)],axis=0) 
                      imagefinal[i][j]=1
              
                      
            
          
                
    currentp=np.delete(currentp,0,axis=0)
    return(currentp,imagefinal)
            
    


def extendedge(currentp,threshold,tfinal):
    for i in currentp:
        m=i[0]
        n=i[1]
        for ii in range(-1,2):
            for jj in range(-1,2):
                if(ii==0 and jj==0):
                    tfinal[m][n]=1
                ti=m+ii
                tj=n+jj
                if(threshold[ti,tj]==1 and  tfinal[ti,tj]==1):
                    tfinal[ti,tj]=1
    
    (h,w)=tfinal.shape
    
    for i in range(h):
        for j in range(w):
            if(tfinal[i][j]==1 ):
                tfinal[i][j]=255
  
    return tfinal
        
        


def thres(image):
    sthre=np.array(image.copy())
    tthre=np.array(image.copy())
    (h,w)=image.shape
    
    for i in range(1,h-1):
        for j in range(1,w-1):
            sthre[i][j]=0
            tthre[i][j]=0
            
            if(image[i][j]>91 ):
                sthre[i][j]=1
                tthre[i][j]=2
            elif(image[i][j]<91 and image[i][j]>20 ):
                tthre[i][j]=1
            
    return(sthre,tthre)
                
    
    
 
    
    
image =np.array(cv2.imread('kf.jpg',cv2.IMREAD_GRAYSCALE))



gaussian_fast = jit(double[:,:](double[:,:]))(gaussian)

gradient_fast = jit(double[:,:](double[:,:]))(gradient)

thres_fast = jit(double[:,:](double[:,:]))(thres)

nonmaxima_fast=jit(double[:,:](double[:,:],double[:,:]))(nonmaxima)

hysteresis_fast=jit(double[:,:](double[:,:],double[:,:]))(hysteresis)

extendedge_fast=jit(double[:,:](double[:],double[:,:],double[:,:]))(extendedge)



start=time.time()

gaussian_out=gaussian_fast(image)

(image_out_grad,image_out_angle)=gradient_fast(gaussian_out) 
 
nonmaxima_img=nonmaxima_fast(image_out_grad,image_out_angle)

(threshold,strong)=thres_fast(nonmaxima_img)

(currentp,image_final)=hysteresis(strong,threshold)

edgeimg=extendedge_fast(currentp,image_final,threshold)

end=time.time()

print(end-start)
 



cv2.imwrite('testout.jpg' , gaussian_out)
cv2.imwrite('gradout.jpg' , image_out_grad)
cv2.imwrite('nonmax.jpg',nonmaxima_img)
cv2.imwrite('final.jpg',edgeimg)



