# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:43:32 2019

@author: wwj
"""


T = int(input())
for t in range(T):
    N = int(input()) #幾筆單
    # list to list
    buy = [] # 價格越高越好
    sell = [] # 價格越低越好
    stockprice = 0
    for n in range(N):
        str = input() #吃整串
        strlist = str.split() #將字串切割 只需要取0,1,4
        share = int(strlist[1])
        price = int(strlist[-1])
        if strlist[0] == 'buy':
            while len(sell)>0:
                #只要賣單還有的話就要跟著比下去(每次比條件最好的那一筆)
                order = sell[0]
                if order[0]>price:
                    # 第一個要跳到的條件
                    # best sell的價格都比我現在要買的價格還要高的化 就不看了
                    break
                dealno = min(share,order[1]) #成交張數
                stockprice = order[0]
                order[1] -= dealno
                share -= dealno
                if order[1] == 0:
                    del sell[0] #如果最上面那個價格已經全數賣出 就刪掉
                if share == 0:
                    break #這次進來的已經全部買了 不用再去sell比對
            if share > 0:
                i = 0
                while i < len(buy) and price < buy[i][0]:
                    i += 1
                if i < len(buy) and price == buy[i][0]:
                    buy[i][1] += share
                else:
                    buy.insert(i,[price,share])
        else:
            while len(buy)>0:
                order = buy[0]
                if order[0]<price:
                    break
                dealno = min(share, order[1])
                #!!!!!!!!!!!!!!!!!
                stockprice = price
                order[1] -= dealno
                share -= dealno
                if order[1] == 0:
                    del buy[0]
                if share == 0:
                    break
            if share > 0:
                i = 0
                while i < len(sell) and price > sell[i][0]:
                    i += 1
                if i < len(sell) and price == sell[i][0]:
                    sell[i][1] += share
                else:
                    sell.insert(i,[price,share])
        if len(sell) == 0:
            s = '-'
        else:
            s = sell[0][0]
        if len(buy) == 0:
            b = '-'
        else:
            b = buy[0][0]
        if stockprice == 0:
            sp = '-'
        else:
            sp = stockprice
        print("%s %s %s"%(s,b,sp))