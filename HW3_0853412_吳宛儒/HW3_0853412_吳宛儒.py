# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:24:58 2019

@author: wwj
"""
import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
features = iris.data[idx,:]
targets = iris.target[idx]

#%%
#越不亂越好
def entropy(p1,n1):
    if (p1 == 0 and n1 == 0):
        return 1
    elif (p1 == 0):
        return 0
    elif (n1 == 0):
        return 0
    pp = p1 / (p1 + n1)
    pn = n1 / (p1 + n1)
    return -pp*math.log2(pp) - pn*math.log2(pn)
#%%
#越大越好 資訊量增加的越多    
def IG(p1,n1,p2,n2):
    num1 = p1 + n1
    num2 = p2 + n2
    num = num1 + num2
    return entropy(p1 + p2,n1 + n2) - (num1/num*entropy(p1,n1) + num2/num*entropy(p2,n2)) 
#%%
def tree1(target,feature,t_target,t_feature):
    node=dict()
    node['data']=range(len(target))
    Tree = [];
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==0),sum(target[G1]==1),sum(target[G2]==0),sum(target[G2]==1))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target[idx]==1)>sum(target[idx]==0)):
                    Tree[t]['decision']=1
                else:
                    Tree[t]['decision']=0
        t+=1
    decision_list = []
    for i in range(len(t_target)):
        test_feature = t_feature[i,:]
        now = 0
        while(Tree[now]['leaf'] == 0):
            if test_feature[Tree[now]['selectf']] <= Tree[now]['threshold']: 
                now = Tree[now]['child'][0]
            else:
                now = Tree[now]['child'][1]
        decision_list.append(Tree[now]['decision'])
    return decision_list 
#%%    
def tree2(target,feature,t_target,t_feature):
    node=dict()
    node['data']=range(len(target))
    Tree = [];
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        elif(sum(target[idx])==2*len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=2
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==1),sum(target[G1]==2),sum(target[G2]==1),sum(target[G2]==2))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target[idx]==1)>sum(target[idx]==2)):
                    Tree[t]['decision']=1
                else:
                    Tree[t]['decision']=2
        t+=1
    decision_list = []
    for i in range(len(t_target)):
        test_feature =  t_feature[i,:]
        now = 0
        while(Tree[now]['leaf'] == 0):
            if test_feature[Tree[now]['selectf']] <= Tree[now]['threshold']: 
                now = Tree[now]['child'][0]
            else:
                now = Tree[now]['child'][1]
        decision_list.append(Tree[now]['decision'])
    return decision_list
#%%
def tree3(target,feature,t_target,t_feature):
    node=dict()
    node['data']=range(len(target))
    Tree = [];
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==2*len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=2
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==0),sum(target[G1]==2),sum(target[G2]==0),sum(target[G2]==2))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target[idx]==0)>sum(target[idx]==2)):
                    Tree[t]['decision']=0
                else:
                    Tree[t]['decision']=2
        t+=1
    decision_list = []
    for i in range(len(t_target)):
        test_feature = t_feature[i,:]
        now = 0
        while(Tree[now]['leaf'] == 0):
            if test_feature[Tree[now]['selectf']] <= Tree[now]['threshold']: 
                now = Tree[now]['child'][0]
            else:
                now = Tree[now]['child'][1]
        decision_list.append(Tree[now]['decision'])
           
    return decision_list

#%%
zero =[]
one = []
two = []
for i in range(len(targets)):
    if targets[i] == 0:
        zero.append(i)
    elif targets[i] == 1:
        one.append(i)
    else:
        two.append(i)
index = []
for i in range(150):
    index.append(i)
    
kf = KFold(n_splits=5, random_state=None, shuffle=False)
kf.get_n_splits(features)
CM = np.zeros((3,3),dtype=int)    
for train_index, test_index in kf.split(features):
    #10
    train_index1 = set(train_index) - set(two)
    train_index1 = list(train_index1)
    d_list1 = tree1(targets[train_index1],features[train_index1],targets[test_index],features[test_index])
    #12
    train_index2 = set(train_index) - set(zero)
    train_index2 = list(train_index2)
    d_list2 = tree2(targets[train_index2],features[train_index2],targets[test_index],features[test_index]) 
    #02
    train_index3 = set(train_index) - set(one)
    train_index3 = list(train_index3)
    d_list3 = tree3(targets[train_index3],features[train_index3],targets[test_index],features[test_index])
    
    test_ans = np.zeros(len(targets[test_index]),dtype=int)
    i = 0
    for j in targets[test_index]:
        test_list = np.zeros(3, dtype=int).tolist()       
        test_list[d_list1[i]] += 1
        test_list[d_list2[i]] += 1
        test_list[d_list3[i]] += 1
        if max(test_list) == 1:
            test_ans[i] = 2
        else:
            test_ans[i] = test_list.index(max(test_list))
        CM[j,test_ans[i]] += 1
        i += 1

print('test acc : ',(CM[0,0]+CM[1,1]+CM[2,2])/150)
print('confusion matrix')
print(CM)