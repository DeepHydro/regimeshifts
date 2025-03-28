# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:58:13 2025

@author: Administrator
"""

import regime_shifts as rs
import ews
import matplotlib.pyplot as plt

ts = rs.sample_rs(std=0.1)

# =============================================================================
# fig, ax = plt.subplots()
# ts.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('System state',fontsize=12)
# plt.show()
# =============================================================================

# =============================================================================
# ts = rs.Regime_shift(ts)
# detection_index = ts.as_detect()
# fig, ax = plt.subplots()
# detection_index.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('Detection Index',fontsize=12);
# plt.show()
# 
# bef_rs = ts.before_rs()
# fig, ax = plt.subplots()
# bef_rs.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('System state',fontsize=12);
# plt.show()
# =============================================================================
 
series = ews.Ews(ts)
series = series.rename(columns={0:'Sample series'}) ## The Ews class returns an extended Dataframe object, if we provided a series, it sets 0 for the column name. 

trend = series.gaussian_det(bW=60).trend
residuals = series.gaussian_det(bW=60).res
fig, axs = plt.subplots(2,1,sharex=True)
ts.plot(ax=axs[0],label='')


trend['Sample series'].plot(ax=axs[0],label='Trend bW=60',linewidth=2)
residuals['Sample series'].plot(ax=axs[1])
axs[1].set_xlabel('Time',fontsize=12)
axs[0].set_ylabel('System state',fontsize=12);
axs[1].set_ylabel('Residuals',fontsize=12);
axs[0].legend(frameon=False);
plt.show()

wL = 200 ## Window length specified in number of points in the series
bW = 60
ar1 = series.ar1(detrend=False,bW=bW,wL=wL) ### Computing lag-1 autocorrelation using the ar1() method
var = series.var(detrend=False,bW=bW,wL=wL) ## Computing variance

series.ar1(detrend=True,bW=bW,wL=wL).kendall
print(f'AR(1) tau = {ar1.kendall:0.3f}')
print(f'Var tau = {var.kendall:0.3f}')

fig, axs = plt.subplots(3,1,sharex=True,figsize=(7,7))
ts.plot(ax=axs[0],legend=False)
ar1['Sample series'].plot(ax=axs[1],label=rf"Kendall's $\tau =$ {ar1.kendall:.2f}")
var['Sample series'].plot(ax=axs[2],label=rf"Kendall's $\tau =$ {var.kendall:.2f}")
axs[0].set_ylabel('System state',fontsize=13)
axs[1].set_ylabel('AR(1)',fontsize=13)
axs[2].set_ylabel('Variance',fontsize=13)
axs[1].legend(frameon=False)
axs[2].legend(frameon=False)
axs[2].set_xlabel('Time',fontsize=13);

pearson = series.pearsonc(detrend=True,bW=bW,wL=wL) ### Computing lag-1 autocorrelation using the pearsonc() method
fig,axs = plt.subplots()
ar1['Sample series'].plot(ax=axs,linewidth=3,label=rf"AR(1) $\tau =$ {ar1.kendall:.2f}")
pearson['Sample series'].plot(ax=axs,label=rf"Pearson corr. $\tau =$ {pearson.kendall:.2f}")
axs.legend(frameon=False)
axs.set_ylabel('Lag-1 autocorrelation',fontsize=13)
axs.set_xlabel('Time',fontsize=13);
plt.show()

# =============================================================================
# rob = series.robustness(indicators=['pearsonc','var'])
# rob['Sample series']['pearsonc']
# rob.plot(vmin=0.1,cmap='viridis')
# =============================================================================
