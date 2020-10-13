# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:59:58 2019

@author: wwj
"""

from sklearn import datasets
iris = datasets.load_iris()
G = iris.data
T = iris.target

#%%

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import Counter

#%%
'''
kmeans
'''
def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D)) #k個中心點的座標
    L = np.zeros((N,1)) #N個人的label
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iteration = 0
    while iteration<maxiter:
        for i in range(K):
            dist[:,i] = np.sum((sample - np.tile(C[i,:],(N,1)))**2,1) 
            #整個迴圈一起做 軸1的方向 #所有距離 #C[i,:]某個座標 乘以(N,1) 等於copy N遍
        L1 = np.argmin(dist,1) #找最小距離
        if iteration>0 and np.array_equal(L,L1):
            #print(iteration)
            break #已經沒有更新嘞
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if len(idx) > 0:
               C[i,:] = np.mean(sample[idx,:],0)  
        wicd = np.sum(np.sqrt(np.sum((sample - C[L,:])**2,1)))
        iteration += 1
    return C,L,wicd #C:幾個中心點 #L:all

#%%
C,L,wicd = kmeans(G,3,1000)
print(wicd)

GA = (G-np.tile(np.mean(G,0),(G.shape[0],1)))/np.tile(np.std(G,0),(G.shape[0],1)) 
C1,L1,wicd1 = kmeans(GA,3,1000)
wicd1 = np.sum(np.sqrt(np.sum((G - C1[L1,:])**2,1)))
#GA L
print(wicd1)

# min max approach
GB = (G-np.tile(np.min(0),(G.shape[0],1))) / (np.tile(np.max(G),(G.shape[0],1))-np.tile(np.min(G),(G.shape[0],1)))
C2,L2,wicd2 = kmeans(GB,3,1000)
wicd2 = np.sum(np.sqrt(np.sum((G - C2[L2,:])**2,1)))
#GB L
print(wicd2)


#%%
'''
knn
'''
def knn(test,train,target,k):
    # test: 1
    # train: 149
    # target: 149
    # k:
    N = train.shape[0]
    dist = np.sum((np.tile(test,(N,1))-train)**2,1)
    idx = sorted(range(len(dist)),key=lambda i: dist[i])[0:k]
    #print(idx)
    return Counter(target[idx]).most_common(1)[0][0]

#%%

CM = np.zeros((3,3))

for j in range(10):
    CM = np.zeros((3,3))
    for i in range(len(G)):
        train = np.delete(G, i, axis = 0)
        test = G[i]
        target = T
        #np.delete(T, i, axis = 0)
        pred = knn(test,train,target,j+1)
        
        CM[T[i],pred] += 1
    print(CM)


