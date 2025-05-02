# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv('Apple Stock Price History.csv',usecols=[0,1,2,3,4])

POHL_avg=f[['Price','Open','High','Low']].mean(axis=1)

period = np.arange(1,len(f)+1,1)

plt.plot(period,POHL_avg,'r',label='My First Plot')