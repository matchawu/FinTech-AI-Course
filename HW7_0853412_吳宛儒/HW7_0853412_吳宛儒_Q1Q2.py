# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:43:20 2019

@author: wwj
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
npzfile = np.load('CBCL.npz')

trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

#%%
fn = 0
ftable = []
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19):
                    fn = fn+1
                    ftable.append([0,y,x,h,w])
print(fn)

for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn+1
                    ftable.append([1,y,x,h,w])
print(fn)
                 
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn+1
                    ftable.append([2,y,x,h,w])
print(fn)                  
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn+1
                    ftable.append([3,y,x,h,w])
print(fn)

#%%

def FeatureExtracting(sample,ftable,c): #取c個特徵
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)
    elif(ftype==1):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y+h:y+h*2,x:x+w].flatten()
        output = np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx1],axis=1)
    elif(ftype==2):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        idx3 = T[y:y+h,x+w*2:x+w*3].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)
    else:
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        idx3 = T[y:y+h*2,x:x+w].flatten()
        idx4 = T[y:y+h*2,x+w*2:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx3],axis=1)+np.sum(sample[:,idx4],axis=1)
    return output


#%%
trpf = np.zeros((trpn,fn)) #2429 x 36648
trnf = np.zeros((trnn,fn)) #4548 x 36648
for c in range(fn):
    trpf[:,c] = FeatureExtracting(trainface,ftable,c)
    trnf[:,c] = FeatureExtracting(trainnonface,ftable,c)

#%%
tepf = np.zeros((tepn,fn)) #2429 x 36648
tenf = np.zeros((tenn,fn)) #4548 x 36648
for c in range(fn):
    tepf[:,c] = FeatureExtracting(testface,ftable,c)
    tenf[:,c] = FeatureExtracting(testnonface,ftable,c)

#%%    
def WeakClassifier(pw,nw,pf,nf):
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
    polarity = 1
    if(error>0.5):
        polarity = 0
        error = 1 - error #反過來做決定比較好
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            polarity = 0
            error = 1 - error #反過來做決定比較好
        if(error<min_error):
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error, min_theta, min_polarity

#%%
 # initialize weights
pw = np.ones((trpn,1))/trpn/2
nw = np.ones((trnn,1))/trnn/2

SC = [] #strong classifier會取的特徵
for t in range(200):
    # normalization
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WeakClassifier(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0 # 假定第0個是最好的特徵
    # try other features, find the best
    for i in range(1,fn):
        me,mt,mp = WeakClassifier(pw,nw,trpf[:,i],trnf[:,i])
        if(me<best_error):
            best_error = me
            best_feature = i
            best_theta = mt
            best_polarity = mp
    beta = best_error/(1-best_error)
    if(best_polarity == 1):
        pw[trpf[:,best_feature]>=best_theta]*=beta
        nw[trnf[:,best_feature]<best_theta]*=beta
    else:
        pw[trpf[:,best_feature]<best_theta]*=beta
        nw[trnf[:,best_feature]>=best_theta]*=beta
    alpha = np.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    print(t)
    print(best_feature)
 
#%%
    
# train
trps = np.zeros((trpn,1))
trns = np.zeros((trnn,1))
alpha_sum = 0
for i in range(100):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        trps[trpf[:,feature]>=theta] += alpha
        trns[trnf[:,feature]>=theta] += alpha
    else:
        trps[trpf[:,feature]<theta] += alpha
        trns[trnf[:,feature]<theta] += alpha
trps /= alpha_sum
trns /= alpha_sum
       
x = []
y = []
for i in range(1000):
    threshold = i/1000
    x.append(np.sum(trns>=threshold)/trnn)
    y.append(np.sum(trps>=threshold)/trpn)

plt.plot(x,y)

#%%
    
# test
teps = np.zeros((tepn,1))
tens = np.zeros((tenn,1))
alpha_sum = 0
for i in range(200):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        teps[tepf[:,feature]>=theta] += alpha
        tens[tenf[:,feature]>=theta] += alpha
    else:
        teps[tepf[:,feature]<theta] += alpha
        tens[tenf[:,feature]<theta] += alpha
teps /= alpha_sum
tens /= alpha_sum
       
x = []
y = []
for i in range(1000):
    threshold = i/1000
    x.append(np.sum(tens>=threshold)/tenn)
    y.append(np.sum(teps>=threshold)/tepn)

plt.plot(x,y)

#%%
mine = me
mini = 0

for i in range(1,fn):
    me,mt,mp = WeakClassifier(pw,nw,trpf[:,i],trnf[:,i])
    if(me<mine):
        mine = me
        mini = i
print([mini,mine])

#%%
#sample is N-by-361 matrix
# return a vector with N feature values
def fe(sample,ftable,c):
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):
        output = 2*sample[:,T[y+h-1,x+w-1].flatten()]+sample[:,T[y,x].flatten()]+sample[:,T[y,x+2*w-1].flatten()]-2*sample[:,T[y,x+w-1].flatten()]-sample[:,T[y+h-1,x].flatten()]-sample[:,T[y+h-1,x+2*w-1].flatten()]
    if(ftype==1):
        output=sample[:,T[y+2*h-1,x+w-1].flatten()]+sample[:,T[y,x+w-1].flatten()]+2*sample[:,T[y+h-1,x].flatten()]-2*sample[:,T[y+h-1,x+w-1].flatten()]-sample[:,T[y+2*h-1,x].flatten()]-sample[:,T[y,x].flatten()]
    if(ftype==2):
        output=sample[:,T[y,x].flatten()]+sample[:,T[y+h-1,x+3*w-1].flatten()]+2*sample[:,T[y,x+2*w-1].flatten()]+2*sample[:,T[y+h-1,x+w-1].flatten()]-sample[:,T[y,x+3*w-1].flatten()]-sample[:,T[y+h-1,x].flatten()]-2*sample[:,T[y,x+w-1].flatten()]-2*sample[:,T[y+h-1,x+2*w-1].flatten()]
    if(ftype==3):
        output = sample[:,T[y,x].flatten()]+sample[:,T[y+2*h-1,x+2*w-1].flatten()]+sample[:,T[y,x+2*w-1].flatten()]+sample[:,T[y+2*h-1,x].flatten()]+4*sample[:,T[y+h-1,x+w-1].flatten()]-2*sample[:,T[y,x+w-1].flatten()]-2*sample[:,T[y+h-1,x].flatten()]-2*sample[:,T[y+h-1,x+2*w-1].flatten()]-2*sample[:,T[y+2*h-1,x+w-1].flatten()]
    return output

#%%
from PIL import Image
#資料測試
bigm=[]
I=np.array(Image.open('test1.jpg').convert('L'))
I2=np.array(Image.open('test1.jpg').convert('L'))
img=np.zeros((19,19))
location=[]
for y in range(0,620,19): # 1080
    for x in range(0,370,19): # 867
        location.append([y,x])
        img=I[x:x+19,y:y+19]
        img2=np.zeros((19,19))
        for i in range(19):
            for j in range(19):
                img2[i,j]=np.sum(img[:i+1,:j+1])
        img2=img2.flatten()
        bigm.append(img2)
bigm=np.array(bigm)
trpf2=np.zeros((100,660))
alpha_sum2=0
trps2=np.zeros(660)
for i in range(100):
    trpf2[i,:]=fe(bigm,ftable,SC[i][0]).reshape(660)
    feature2 = SC[i][0]
    theta2 = SC[i][1]
    polarity2 = SC[i][2]
    alpha2 = SC[i][3]
    alpha_sum2 = alpha_sum2 + alpha2
    if(polarity2==1):
        trps2[trpf2[i,:]>=theta2] = trps2[trpf2[i,:]>=theta2]+alpha2
    else:
        trps2[trpf2[i,:]<theta2] = trps2[trpf2[i,:]<theta2]+alpha2
trps2 = trps2/alpha_sum2
for i in range(660):
    if(trps2[i]>0.5):
        x=location[i][1]
        y=location[i][0]
        I2[x:x+19,y:y+2]=255
        I2[x:x+19,y+17:y+19]=255
        I2[x:x+2,y:y+19]=255
        I2[x+17:x+19,y:y+19]=255
new_im = Image.fromarray(I2)
new_im.show()












































