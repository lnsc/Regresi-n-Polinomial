# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as mt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import array
f = open("data.txt")
data = f.read().strip()
f.close()
 
m = [[int(num) for num in line.strip().split()] for line in data.split('\n')]
print (m)
print (len(m))
print (len(m[0]))
print ("\n")
for x in range(0,len(m[0])):
    print (m[0][x])

theta =  [i for i in range(len(m[0])+1)]

for i in range(0,len(m[0])+1):
    theta[i]=0

def h(x,thet):
    res=0
    for i in range(0,len(m[0])):
        if i== 0:
            res = res + theta[i]
        else:
            res = res + x[i-1]*theta[i] + pow(x[i-1],2)*theta[i+1]
    return res

def funcionCosto(x,t):
    r=0.00
    for i in range(0,len(x)):
        xe=[i for i in range(len(x[0])-1)]
        for j in range(0,len(x[0])-1):
            xe[j]=x[i][j]
        r = r + pow(h(xe,t)-x[i][len(x[0])-1],2)
    r = r/(len(m)*2)
    return r
############
alpha = 0.005

def gradiente(t,x,alpha):
    res2=[i for i in range(len(t))]
    for i in range(0,len(t)):
        res2[i]=t[i]
        
    for j in range(0,len(t)):
        xe=[o for o in range(len(x[0])-1)]
        if j==0:
            t[j]=res2[j]-alpha*funcionCosto(x,res2)/len(x)
        else:
            sumatemp=0.00
            for k in range(len(x)):
                for u in range(0,len(x[0])-1):
                    xe[u]=x[k][u]
                    print("->",xe[u])
                sumatemp=sumatemp + pow(h(xe,t)-x[k][len(x[0])-1],1)*x[k][j-1]
            t[j]=res2[j]-alpha*(sumatemp/len(x))
    print(t)
    
for i in range(1):
    gradiente(theta,m,alpha)      
