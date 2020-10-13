# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:48:08 2019

@author: wwj
"""

import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

#%%
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

I = Image.fromarray(trainface[0,:].reshape((19,19))) #size:1*361
I.show()
#%%
raw = np.zeros((40*19,50*19))
for y in range(40):
    for x in range(50):
        I1 = trainface[y*50+x].reshape((19,19))
        raw[y*19:y*19+19,x*19:x*19+19] = I1

I = Image.fromarray(raw)
I.show()
#%%
def BPNNtrain(pf,nf,hn,lr,iteration):
    # pf : 所有正資料 是臉
    # hn : hidden層有幾個node(節點)(這邊只有一層)
    # lr : learning rate 大:舊的不忘較多 新的更新較少
    # iteration : 要跑多少次
    pn = pf.shape[0] #有多少positive samples
    nn = nf.shape[0]
    fn = pf.shape[1]
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis = 0)
    WI = np.random.normal(0,1,(fn+1,hn))
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        print(t)
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1) #input signal
            oh = ins.dot(WI)
            oh = 1/(1+np.exp(-oh))
            hs = np.append(oh,1)
            out = hs.dot(WO)
            out = 1/(1+np.exp(-out))
            dk = out*(1-out)*(target[s[i]]-out)
            dh = oh*(1-oh)*WO[:hn,0]*dk
            WO[:,0]+=lr*dk*hs
            for j in range(hn):
                WI[:,j]+= lr*dh[j]*ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO
    return model

#%%
def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    hn = WI.shape[1]
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        oh = ins.dot(WI)
        oh = 1/(1+np.exp(-oh))
        hs = np.append(oh,1)
        out[i] = hs.dot(WO)
        out[i] = 1/(1+np.exp(-out[i]))
    return out
        
    
network = BPNNtrain(trainface/255,trainnonface/255,20,0.01,10)
pscore = BPNNtest(trainface/255,network)
nscore = BPNNtest(trainnonface/255,network)

#%%
X = np.zeros((99,1))
Y = np.zeros((99,1))
for i in range(99):
    threshold = (i+1)/100
    X[i] = np.mean(nscore>threshold) #false alarm weight
    Y[i] = np.mean(pscore>threshold)
plt.plot(X,Y)

#把pscore和nscore改成test就可以看到test的ROC curve