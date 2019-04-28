import numpy as np
import cv2
from numba import jit,double,cuda
import time
import math


limit = 30#input("Input image threshold for edge detector(Higher=less sensitive):")
int_limit = int(limit)




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
        for ii in range(hf):
            for jj in range(wf):
                tsum=tsum+(image[i-hf2+ii,j-wf2+jj]*fil[hf-1-ii,wf-1-jj])
        
        image_out[i][j]=tsum
        
    return image_out

@cuda.jit
def gaussian_cuda(image, fil, gaussian_out):
    (h,w) = image.shape
    (hf,wf)=fil.shape
   
    hf2=hf//2
    wf2=wf//2
    i, j = cuda.grid(2)
    if hf2 <= i and i<= h-hf2 and wf2<=j and j<=w-wf2:
        tsum=0
        for ii in range(hf):
            for jj in range(wf):
                tsum=tsum+(image[i-hf2+ii,j-wf2+jj]*fil[hf-1-ii,wf-1-jj])
        
        gaussian_out[i][j]=tsum
    





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

@cuda.jit
def gradient_cuda(image, image_out2, image_out3, gx, gy):
    # image_out2 = np.array(image.copy())
    # image_out3 = np.array(image.copy())
    
#    gx=([[-1,0,1],[-2,0,2],[-1,0,1]])
#    gy=([[-1,0,1] [-2,0,2] [-1,0,1]])

    (h,w) = image.shape
    (hf,wf)=(3,3)
   
    hf2=hf//2
    wf2=wf//2
    i, j = cuda.grid(2)
    if hf2 < i and i< h-hf2 and wf2<j and j<w-wf2:
        tsumx=0
        tsumy=0
        for ii in range(hf):
            for jj in range(wf):
                tsumx=tsumx+(image[i-hf2+ii,j-wf2+jj]*gx[hf-1-ii,wf-1-jj])
                tsumy=tsumy+(image[i-hf2+ii,j-wf2+jj]*gy[hf-1-ii,wf-1-jj])
        
                
        image_out2[i][j]= ((tsumx*tsumx)+(tsumy*tsumy))**0.5
        theta = math.atan2(tsumy, tsumx)
        image_out3[i][j] =(math.ceil(theta * (5.0 / 3.1415)) + 5) % 5   #angle quantization 


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

@cuda.jit
def nonmaxima_cuda(image,imagea,image_out_nmax):
    (h,w)=image.shape
    i, j = cuda.grid(2)
    if 0 <= i and i< h and 0<=j and j<w:   
            if(i==0 or i==h-1 or j==0 or j==w-1 ):
                image_out_nmax[i][j]=0
                
            
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
    
            
    



def hysteresis(imageS,imageT):
    imagefinal = np.array(imageS.copy())
    imageth=np.array(imageT.copy())
    currentp=np.array([(-1,-1)])
    (h,w) = image.shape
    
    for i in range(h):
        for j in range(w):
            if(imageth[i][j]!=1):
                continue
            
            window=imageth[i-1:i+2,j-1:j+2]
            wmax=window.max()
            if(wmax==2):
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
            
            if(image[i][j]>int_limit ):
                sthre[i][j]=1
                tthre[i][j]=2
            elif(image[i][j]<int_limit and image[i][j]>20 ):
                tthre[i][j]=1
            
    return(sthre,tthre)
                
    
    
image =np.array(cv2.imread('400k.jpg',cv2.IMREAD_GRAYSCALE))
 
 # Create the data array - usually initialized some other way
threadsperblock = (32, 32)
blockspergrid_x = int(math.ceil(image.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(image.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

gaussian_fast = jit(double[:,:](double[:,:]))(gaussian)


gradient_fast = jit(double[:,:](double[:,:]))(gradient)

thres_fast = jit(double[:,:](double[:,:]))(thres)

nonmaxima_fast=jit(double[:,:](double[:,:],double[:,:]))(nonmaxima)

hysteresis_fast=jit(double[:,:](double[:,:],double[:,:]))(hysteresis)

extendedge_fast=jit(double[:,:](double[:],double[:,:],double[:,:]))(extendedge)



start=time.time()
###############################################################################
## CUDA GPU calculation
###############################################################################
#Copy imgae to GPU memory
dev_image = cuda.to_device(image)
#Allocate device memory for gaussian output
dev_gaussian_out    = cuda.device_array(image.shape)
#CUDA gaussian kernel
gaussian_cuda[blockspergrid, threadsperblock](dev_image, 
              np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]),
              dev_gaussian_out)
#Allocate device memory for gradient output
dev_image_out_grad  = cuda.device_array(image.shape)
dev_image_out_angle = cuda.device_array(image.shape)
#CUDA gradient kernel
gradient_cuda[blockspergrid, threadsperblock](dev_gaussian_out, dev_image_out_grad,
             dev_image_out_angle, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
             np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
#Transfer output to host
#cuda.synchronize()
image_out_grad = dev_image_out_grad.copy_to_host()
image_out_angle = dev_image_out_angle.copy_to_host()

#dev_image_out_nmax = cuda.device_array(image.shape)
#nonmaxima_cuda[blockspergrid, threadsperblock](dev_image_out_grad,
#             dev_image_out_angle, 
#             dev_image_out_nmax)
#nonmaxima_img = dev_image_out_nmax.copy_to_host()
#
end=time.time()

print(end-start)

nonmaxima_img=nonmaxima_fast(image_out_grad,image_out_angle)

(threshold,strong)=thres_fast(nonmaxima_img)

(currentp,image_final)=hysteresis_fast(strong,threshold)

edgeimg=extendedge_fast(currentp,image_final,threshold)

end2=time.time()
#
print(end2-start)
###############################################################################

#gaussian_out=gaussian_fast(image)
#
#(image_out_grad,image_out_angle)=gradient_fast(gaussian_out) 
#
#end=time.time()
#
#print(end-start)
#
#nonmaxima_img=nonmaxima_fast(image_out_grad,image_out_angle)
#
#(threshold,strong)=thres_fast(nonmaxima_img)
#
#(currentp,image_final)=hysteresis_fast(strong,threshold)
#
#edgeimg=extendedge_fast(currentp,image_final,threshold)
#
##end=time.time()
##
##print(end-start)
 



#cv2.imwrite('gaussianoutput.jpg' , gaussian_out)
cv2.imwrite('output.jpg' , image_out_grad)
cv2.imwrite('nonmaxima.jpg',nonmaxima_img)
cv2.imwrite('Final.jpg',edgeimg)
