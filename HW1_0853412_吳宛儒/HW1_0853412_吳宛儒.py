# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:40:34 2019

@author: wwj
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

call = {10700:266, 
        10800:189, 
        10900:120, 
        11000:67, 
        11100:32, 
        11200:13, 
        11300:6.1}

put = {10700:43, 
       10800:63, 
       10900:93, 
       11000:139, 
       11100:207,
       11200:290,
       11300:360}

#%%
x = np.arange(10500,11501)

def callr(K):
    global x # a series of St
    global call
    #max(ST-K,0)-call
    return np.maximum(x-K, 0) - call[K]

def putr(K):
    global x # a series of St
    global put
    #max(K-ST,0)-put
    return np.maximum(K-x, 0) - put[K]

#%%
'''
1. 用履約價為10900、11000、11100的買權及賣權，共可排列組合出幾種不同的bull spread，
請分別用不同顏色的線畫出所有bull spread的損益曲線，試比較不同到期價時的優缺點
'''

#不同的bull's spread組合
y1 = putr(10900)-putr(11100) #
y2 = putr(11000)-putr(11100) #
y3 = putr(10900)-putr(11000) #
y4 = callr(10900)-callr(11100) #
y5 = callr(11000)-callr(11100) #
y6 = callr(10900)-callr(11000) #
plt.plot(x, y1, 'r', x, y2, 'g', x, y3, 'b', x, y4, 'y', x, y5, 'cyan', x, y6, 'grey',[x[0],x[-1]],[0,0],'--k')
plt.legend(['put109-111','put110-111','put109-110','call109-111','call110-111','call109-110'])

#%%
'''
2. 假設加權股價持續盤整，直到到期日前均會在11000附近震盪，
A.	使用履約價為11000的買賣權建構straddle
B.	使用履約價為10800和11200的買賣權建構strangle
分別用紅線及綠線繪製兩者的損益曲線，並比較兩者的優缺點

'''
#straddle 賣出
ys1 = -callr(11000) #
ys2 = -putr(11000) #
ys3 = ys1+ys2

#strangle 賣出
ys4 = -callr(11200) #
ys5 = -putr(10800) #
ys6 = ys4+ys5
plt.plot(x, ys3, 'r', x, ys6, 'g',[x[0],x[-1]],[0,0],'--k')
plt.legend(['straddle','strangle'])

#%%
'''
3. 請用不同履約價的買權，組出兩個預期市場盤整時的butterfly spread，並簡述兩者適用的情境
'''
#butterfly spread
bs1 = callr(10800) #
bs2 = callr(11200) #
bs3 = -callr(11000)*2
bs4 = bs1+bs2+bs3

bs5 = callr(11000) #
bs6 = callr(11200) #
bs7 = -callr(11100)*2
bs8 = bs5+bs6+bs7
plt.plot(x, bs4, 'r', x, bs8, 'g',[x[0],x[-1]],[0,0],'--k')




