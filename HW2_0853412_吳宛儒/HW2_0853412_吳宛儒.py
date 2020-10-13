# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:49:34 2019

@author: wwj
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

#%%
'''
monte-carlo methods
'''
def MCsim(S,T,r,vol,N):
    #模擬股價走勢
    dt = T / N
    #St = np.zeros((N+1))
    global St
    St[0] = S
    for i in range(N):
        St[i+1] = St[i]*math.exp((r-0.5*vol*vol)*dt + \
          np.random.normal()*vol*math.sqrt(dt))
    return St

# option 定價需要的參數
S = 50 #定價
L = 40 #履約價
T = 2 #距離到期年數
r = 0.08 #無風險利率(台灣：定存；美國：長年期債券)
vol = 0.2 #sigma波動率，為定價的人的參數，由估計得出
N = 100

St = np.zeros((N+1))

Sa = MCsim(S,T,r,vol,N)
plt.plot(Sa)

#%%
M = 10000
call = 0
for i in range(M):
    Sa = MCsim(S,T,r,vol,N)
    #plt.plot(Sa)
    if(Sa[-1]-L>0):
        call += Sa[-1]-L
print(call/M*math.exp(-r*T))
#plt.show()

#%%
'''
Black Scholes Model
'''
def BLSprice(S,L,T,r,vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    #norm.cdf(x) 負無限大到x的累積函數
    call = S * norm.cdf(d1) - L*math.exp(-r*T)*norm.cdf(d2)
    return call

#%%
print(BLSprice(S,L,T,r,vol))

#%%
'''
Binomial tree
'''
N = 10000
def BTcall(S,T,r0,vol,N,L):
    dt = T / N
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    p = (math.exp(r0*dt)-d)/(u-d)
    print(u,d,p)
    priceT = np.zeros((N+1,N+1)) #build price tree
    priceT[0][0] = S
    for c in range(N):
        #column 由c去推c+1
        priceT[0][c+1] = priceT[0][c]*u
        for r in range(N):
            priceT[r+1][c+1] = priceT[r][c]*d
    probT = np.zeros((N+1,N+1))
    probT[0][0] = 1
    for c in range(N):
        for r in range(N):
            probT[r][c+1] += probT[r][c]*p
            probT[r+1][c+1] += probT[r][c]*(1-p)
    call = 0
    for r in range(N+1):
        if(priceT[r][N]>=L):
            call += (priceT[r][N]-L)*probT[r][N] #機率*獲利
    return call*math.exp(-r0*T)
    
#%%
print(BTcall(S,T,r,vol,N,L))
    
#%%
call = BLSprice(S,L,T,r,vol)
x = np.zeros((100,1))
y = np.zeros((100,1))
for i in range(100):
    #0~99
    x[i] = i/100
    y[i] = BLSprice(S,L,T,r,x[i])-call    
plt.plot(x,y,'r',[0,1],[0,0],'--k')

#%%
'''
Bi-section method
'''
def BisectionBLS(S,L,T,r,call,tol):
    left = 0.00000000000001
    right = 1
    while(right-left>tol):
        middle = (left+right)/2
        if((BLSprice(S,L,T,r,middle)-call)*(BLSprice(S,L,T,r,left)-call)<0):
            #異號
            right = middle
        else:
            left = middle
    return (left+right)/2

#%%
print(BisectionBLS(S,L,T,r,call,0.00001))


#%%
'''
1.使用 Monte Carlo methods 計算買權價格，
假設 S=50，L=40，T=2，r=0.08，σ=0.2，
請設計一個實驗，嘗試在各種不同切分期數與模擬次數的組合下，
觀察Monte Carlo methods與black-scholes model的絕對誤差的變化，
並解釋實驗結果的合理性。
'''
N = 50000
M = 1000

St = np.zeros((N+1))

Sa = MCsim(S,T,r,vol,N)
plt.plot(Sa)

call = 0
for i in range(M):
    Sa = MCsim(S,T,r,vol,N)
    #plt.plot(Sa)
    if(Sa[-1]-L>0):
        call += Sa[-1]-L
print(call/M*math.exp(-r*T))

#%%
'''
2.建構n層Binomial Tree計算買權價格，請設計一個實驗，嘗試不同的n，
觀察決策樹與black-scholes model的絕對誤差的變化，並解釋實驗結果的合理性。
'''
N = 10000
print(BTcall(S,T,r,vol,N,L))

#%%
'''
3.嘗試用今天上課的選擇權價格，是否能畫出波動率微笑曲線呢
'''
S = 10889.96 

L_today = [10500,10600,10700,10750,
           10800,10850,10900,10950,
           11000,11050,11100,11150,
           11200,11250,11300,11400,
           11500]
call_today = [423,324,233,189,
              152,115,83,55,
              33,18,9.5,4.9,
              2.3,1.2,0.9,0.3,
              0.3]

T = (16-9)/365 #距離到期年數
r = 0.08 #無風險利率(台灣：定存；美國：長年期債券)

ans = np.zeros((17,1))
for i in range(len(L_today)):
    ans[i] = BisectionBLS(S,L_today[i],T,r,call_today[i],0.0000001)
    
plt.plot(ans)





