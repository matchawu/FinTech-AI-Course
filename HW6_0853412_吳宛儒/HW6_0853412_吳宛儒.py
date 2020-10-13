# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:56:10 2019

@author: wwj
"""

import numpy as np
from PIL import Image
from scipy import signal

def loadImage(filename):
    I = Image.open(filename)
    W,H = I.size
    data = np.asanyarray(I)
    return I,W,H,data
filename = 'carol.png'
I,W,H,data = loadImage(filename)
#I = Image.open('carol.png')
#I = Image.open('rouge.jpg')
#I = Image.open('j.jpg')
#W,H = I.size
#data = np.asanyarray(I)

#%%
'''
film & grayscale
'''
I,W,H,data = loadImage(filename)
#data = 255-data #film
data2 = data.copy()
data = data.astype('float64')
gray = (data[:,:,0]+data[:,:,1]+data[:,:,2])/3
data2[:,:,0] = gray
data2[:,:,1] = gray
data2[:,:,2] = gray
I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%
'''
1.
noise
'''
I,W,H,data = loadImage(filename)
data2 = data.copy()
data = data.astype('float64')
noise = np.random.normal(0,10,(H,W,3))

data3 = data+noise
data3[data3>255]=255
data3[data3<0]=0
data2 = data3.astype('uint8')
I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%
'''
2. 加過雜訊的用較小的gaussian去抹掉雜訊
平滑化 Gaussian Smooth
雜點消失
細節消失
'''
I,W,H,data = loadImage(filename)
#M = np.ones((20,20))/400
x,y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.2, 0.0
M = np.exp(-((d-mu)**2/(2.0*sigma**2)))
M = M/np.sum(M[:])
R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]
R2 = signal.convolve2d(R,M, boundary='symm', mode='same')
G2 = signal.convolve2d(G,M, boundary='symm', mode='same')
B2 = signal.convolve2d(B,M, boundary='symm', mode='same')

data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%
'''
3.
用大一點的gaussian去坐失焦圖
會有圓形的感覺 比較像照片失焦
'''
I,W,H,data = loadImage(filename)
#M = np.ones((20,20))/400
x,y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
M = np.exp(-((d-mu)**2/(2.0*sigma**2)))
M = M/np.sum(M[:])
R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]
R2 = signal.convolve2d(R,M, boundary='symm', mode='same')
G2 = signal.convolve2d(G,M, boundary='symm', mode='same')
B2 = signal.convolve2d(B,M, boundary='symm', mode='same')

data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%
'''
4.
有負有正的distribution去做銳利化
'''
filename = 'wall.jpg'
I,W,H,data = loadImage(filename)

data2 = data.copy()
data = data.astype('float64')

Mx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
My = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

gray = (data[:,:,0]+data[:,:,1]+data[:,:,2])/3
#data2[:,:,0] = gray
#data2[:,:,1] = gray
#data2[:,:,2] = gray
#I2 = Image.fromarray(data2,'RGB')
#I2.show()

Ix = signal.convolve2d(gray,Mx, boundary='symm', mode='same')
Iy = signal.convolve2d(gray,My, boundary='symm', mode='same')

I_all = Ix**2+Iy**2

temp = I_all.flatten()
temp = np.sort(temp)
bound = int(len(temp)*0.8)
bound = temp[bound]

I_all[I_all>=bound] = 255
I_all[I_all<255] = 0

data2[:,:,0] = I_all
data2[:,:,1] = I_all
data2[:,:,2] = I_all

I2 = Image.fromarray(data2,'RGB')
I2.show()



