# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:37:09 2019

"""
# Homework 5
#%%
import numpy as np
import matplotlib.pyplot as plt
import math

#%%
def func(t, A, B, C, tc, beta, w, phi):
    dt = tc - t
    inc = dt**beta
    return A + B * inc + B * C * inc * np.cos(w * np.log(dt) + phi)

def error(t, b, A, B, C, tc, beta, w, phi):
    return np.sum(abs(b - func(t, A, B, C, tc, beta, w, phi)))

def get_matA(t, tc, beta, w, phi):
    t = t.reshape(-1, 1)
    dt = tc - t
    inc = dt**beta
    return np.concatenate((np.ones(t.shape),
                           inc,
                           inc * np.cos(w * np.log(dt) + phi)), axis = 1)

#%%
data = np.load('data.npy')

#%%
# initialize A, B, C
A = 1
B = -1
C = 1

pop_n = 10000
gene_n = 34
crossover = 100

# constants
exp = np.arange(10)
t_list = np.arange(data.shape[0])

#%%
# first generation
pop = np.random.randint(0, 2, (pop_n, gene_n))
# initialize fitness
fit = np.zeros((pop_n,))

#%%
for routine in range(10):
    print(f'Routine {routine}')
    
    for generation in range(10):
        # determine fitness
        print(f'Generation {generation}')
        for i in range(pop_n):
            gene = pop[i]
            # tc: [0, 16) + 1151
            tc = (np.sum(2**exp[:4] * gene[0:4])) + 1151
            # beta: (0, 1)
            beta = (np.sum(2**exp * gene[4:14]) + 1) / 1025
            # w: [log2(2) + 1, log2(1025) + 1]
            w = math.log2(np.sum(2**exp * gene[14:24]) + 2) + 1
            # phi: (0, 2pi)
            phi = (np.sum(2**exp * gene[24:]) + 1) / 1025 * 2 * math.pi
            t = t_list[:tc]
            b = np.log(data[t])
            fit[i] = error(t, b, A, B, C, tc, beta, w, phi)
            
        sort_fix = np.argsort(fit)
        pop = pop[sort_fix]
        # crossover
        for i in range(crossover, pop_n):
            fid = np.random.randint(0, crossover)
            mid = np.random.randint(0, crossover)
            while mid == fid:
                mid = np.random.randint(0, crossover)
            mask = np.random.randint(0, 2, (gene_n,))
            child = pop[mid]
            child[mask == 1] = pop[fid][mask == 1]
            pop[i] = child
        # mutation
        for i in range(10000):
            # 第一名不突變，用來做 linear regression
            m = np.random.randint(1, pop_n)
            n = np.random.randint(0, gene_n)
            pop[m, n] = 1 - pop[m, n]
            
    # retrieve best parameters
    gene = pop[0]
    tc = (np.sum(2**exp[:4] * gene[0:4])) + 1151
    beta = (np.sum(2**exp * gene[4:14]) + 1) / 1025
    w = math.log2(np.sum(2**exp * gene[14:24]) + 2) + 1
    phi = (np.sum(2**exp * gene[24:]) + 1) / 1025 * 2 * math.pi
    t = t_list[:tc]
    b = np.log(data[t])
    
    # linear regression
    matA = get_matA(t, tc, beta, w, phi)
    vecX = np.linalg.lstsq(matA, b)[0]
    A = vecX[0]
    B = vecX[1]
    C = vecX[2] / B
    
    print(f'tc = {tc}, beta = {beta}, w = {w}, phi = {phi}')
    print(f'A = {A}, B = {B}, C = {C}')
    pred = func(t_list, A, B, C, tc, beta, w, phi)
    plt.plot(t_list, data, 'b',
             t_list, np.exp(pred), 'r')
    plt.show()