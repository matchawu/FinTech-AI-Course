# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:11:52 2019

@author: wwj
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

def show(profit): #給每天計算完的profit
    profit = np.array(profit)
    profit2 = np.cumsum(profit)
    plt.plot(profit2)
    plt.show()
    
    ans1 = profit2[-1] #總損益點數
    ans2 = np.sum(profit>0)/len(profit) #勝率
    ans3 = np.mean(profit[profit>0])#賺錢時 平均獲利點數
    ans4 = np.mean(profit[profit<=0]) #沒賺錢也當作虧
    plt.hist(profit,bins=100)
    plt.show()
    print('Total:',ans1,'\n Win Ratio:',ans2,'\n Win Average:',ans3,'\n Lose Average:',ans4)
    
#%%

df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values #台指期
tradeday = list(set(TAIEX[:,0]//10000)) #同一天的會放在同一個list內
tradeday.sort()

#%%
# 0.0
profit = np.zeros((len(tradeday),1)) #獲利 每天
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[-1],1] - TAIEX[idx[0],2]
#profit2 = np.cumsum(profit) #連續投資下的獲利情況(累計獲利)
#plt.plot(profit2) #一點一元的話會賺多少錢
show(profit)

#%%
# 0.1
profit = np.zeros((len(tradeday),1)) #獲利 每天
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[0],2] - TAIEX[idx[-1],1]  
#profit2 = np.cumsum(profit) #連續投資下的獲利情況(累計獲利)
#plt.plot(profit2) #一點一元的話會賺多少錢
show(profit)

#%%
'''
策略1.0
開盤買進一口，30點停損，收盤平倉
策略1.1
開盤空一口，30點停損，收盤平倉
'''

#1.0

profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #某幾分鐘的最低價(4)觸及p1-30
    if(len(idx2)==0):
        p2 = TAIEX[idx[-1],1]
    else:
        p2 = TAIEX[idx[idx2[0]],1]
    profit.append(p2-p1)
show(profit)

#%%
#1.1

profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+30)[0] #某幾分鐘的最低價(4)觸及p1+30
    if(len(idx2)==0):
        p2 = TAIEX[idx[-1],1]
    else:
        p2 = TAIEX[idx[idx2[0]],1]
    profit.append(p1-p2)
show(profit)


#%%
'''
策略2.0
開盤買進一口，30點停損，30點停利，收盤平倉
策略2.1
開盤空一口，30點停損，30點停利，收盤平倉
'''

#2.0
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #某幾分鐘的最低價(4)觸及p1-30 #停損
    idx3 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #停利
    if(len(idx2)==0 and len(idx3)==0): #沒有停損也沒有停利:用當天的收盤價買
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0):
        p2 = TAIEX[idx[idx2[0]],1] #遇到停損點那一分鐘的價錢賣
    elif(len(idx2)==0):
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1] 
        
    profit.append(p2-p1)
show(profit)

#%%
#2.1
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+30)[0] #某幾分鐘的最低價(4)觸及p1-30 #停損
    idx3 = np.nonzero(TAIEX[idx,3]<=p1-30)[0] #停利
    if(len(idx2)==0 and len(idx3)==0): #沒有停損也沒有停利:用當天的收盤價買
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0):
        p2 = TAIEX[idx[idx2[0]],1] #遇到停損點那一分鐘的價錢賣
    elif(len(idx2)==0):
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1] 
        
    profit.append(p1-p2)
show(profit)


#%%
'''
策略3.0
找到一組m, n改進策略2.0使總損益點數最佳化
策略3.1
找到一組m, n改進策略2.1使總損益點數最佳化
'''

#3.0

total = []
for u in range(10):
    N = (u+1)*10
    for k in range(10):
        M = (k+1)*10
        profit = []
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
            idx.sort()
            p1 = TAIEX[idx[0],2]
            idx2 = np.nonzero(TAIEX[idx,4]<=p1-N)[0] #某幾分鐘的最低價(4)觸及p1-30 #停損 n
            idx3 = np.nonzero(TAIEX[idx,3]>=p1+M)[0] #停利 m
            #print(idx2,idx3)
            if(len(idx2)==0 and len(idx3)==0): #沒有停損也沒有停利:用當天的收盤價買
                p2 = TAIEX[idx[-1],1]
            elif(len(idx3)==0):
                p2 = TAIEX[idx[idx2[0]],1] #遇到停損點那一分鐘的價錢賣
            elif(len(idx2)==0):
                p2 = TAIEX[idx[idx3[0]],1]
            elif idx2[0]<idx3[0]:
                p2 = TAIEX[idx[idx2[0]],1]
            else:
                p2 = TAIEX[idx[idx3[0]],1] 
            profit.append(p2-p1)
        #show(profit)
        
        profit = np.array(profit)
        profit2 = np.cumsum(profit)
        
        ans1 = profit2[-1]
        total.append(ans1)
    
    print(total)
print('MAX:',max(total),'index',total.index(max(total)))  

#%%
# 瞭解到最大為m=60,n=20，印出其profit
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-20)[0] #某幾分鐘的最低價(4)觸及p1-30 #停損
    idx3 = np.nonzero(TAIEX[idx,3]>=p1+60)[0] #停利
    if(len(idx2)==0 and len(idx3)==0): #沒有停損也沒有停利:用當天的收盤價買
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0):
        p2 = TAIEX[idx[idx2[0]],1] #遇到停損點那一分鐘的價錢賣
    elif(len(idx2)==0):
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1] 
        
    profit.append(p2-p1)
show(profit)

#%%

#3.1

total = []
for u in range(10):
    N = (u+1)*10
    for k in range(10):
        M = (k+1)*10
        profit = []
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
            idx.sort()
            p1 = TAIEX[idx[0],2]
            idx2 = np.nonzero(TAIEX[idx,4]>=p1+N)[0] #某幾分鐘的最低價(4)觸及p1-30 #停損 n
            idx3 = np.nonzero(TAIEX[idx,3]<=p1-M)[0] #停利 m
            #print(idx2,idx3)
            if(len(idx2)==0 and len(idx3)==0): #沒有停損也沒有停利:用當天的收盤價買
                p2 = TAIEX[idx[-1],1]
            elif(len(idx3)==0):
                p2 = TAIEX[idx[idx2[0]],1] #遇到停損點那一分鐘的價錢賣
            elif(len(idx2)==0):
                p2 = TAIEX[idx[idx3[0]],1]
            elif idx2[0]<idx3[0]:
                p2 = TAIEX[idx[idx2[0]],1]
            else:
                p2 = TAIEX[idx[idx3[0]],1] 
            profit.append(p1-p2)
        #show(profit)
        
        profit = np.array(profit)
        profit2 = np.cumsum(profit)
        
        ans1 = profit2[-1]
        total.append(ans1)
    
    print(total)
print('MAX:',max(total),'index',total.index(max(total)))  


#%%
# 瞭解到最大為m=90,n=10，印出其profit
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+10)[0] #某幾分鐘的最低價(4)觸及p1-30 #停損
    idx3 = np.nonzero(TAIEX[idx,3]<=p1-90)[0] #停利
    if(len(idx2)==0 and len(idx3)==0): #沒有停損也沒有停利:用當天的收盤價買
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0):
        p2 = TAIEX[idx[idx2[0]],1] #遇到停損點那一分鐘的價錢賣
    elif(len(idx2)==0):
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1] 
        
    profit.append(p1-p2)
show(profit)














