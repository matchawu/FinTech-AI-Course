# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:36:06 2019

@author: wwj
"""

#r=0.1
#n=1~10000
#%%
import matplotlib.pyplot as plt
#%%
y=[]
p0 = 100

#%%
for m in range(1,10001):
    y.append(p0*(1+0.1/m)**m)
plt.plot(y[:20])

#%%
import math
p0*math.exp(0.1)

#%%
import pandas as pd

data = pd.read_csv("KO.csv")

#%%
plt.plot(data['Adj Close'],color='grey')
#能賺的錢：55/35-1
#%%
data['return'] = (data['Close'] - data['Open'])/data['Close']

#%%
for i in range(len(data)):
    if i = len(data) - 1:
        data['return'][i] = data['Close'][i]
    else:
        data['return'][i] = (data['Close'][i+1]-data['Close'][i])/data['Close'][i]

#%%
plt.hist(data['return'])
#plt.plot(data['Close'])
#plt.plot(data['Open'])
#plt.plot(data['High'])
#plt.plot(data['Low'])

