import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur
import os


def make_plot():
    df=pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv",encoding="gbk")
    y=df.groupby('IGBP').b.mean()
    y=pd.DataFrame(y)
    sem=df.groupby('IGBP').b.sem()
    sem=pd.DataFrame(sem)
    y.insert(1,'sem_all',sem.b)
    
    #计算北半球不同植被的b值和标准误
    y_N=df[df['latitude']>0].groupby('IGBP').b.mean()
    y_N=pd.DataFrame(y_N)
    y.insert(1,'b_N',y_N.b)
    sem_N=df[df['latitude']>0].groupby('IGBP').b.sem()
    sem_N=pd.DataFrame(sem_N)
    y.insert(1,'sem_N',sem_N.b)
    
    #计算南半球不同植被的b值和标准误
    y_S=df[df['latitude']<0].groupby('IGBP').b.mean()
    y_S=pd.DataFrame(y_S)
    y.insert(1,'b_S',y_S.b)
    sem_S=df[df['latitude']<0].groupby('IGBP').b.sem()
    sem_S=pd.DataFrame(sem_S)
    y.insert(1,'sem_S',sem_S.b)
    
    #加入类型
    type=['CRO','CSH','DBF','DNF','EBF','ENF','GRA','MF','OSH','SAV','WET','WSA']
    y.insert(3,'type',type)
    y.to_csv("Fluxnet/a_folder/各植被Bowen.csv")
    
    #调整顺序
    order=[12,6,4,3,2,1,10,5,7,9,11,8]
    y.insert(3,'order',order)
    y.sort_values(by='order',ascending=True, inplace=True) 
    
    #绘图
    fig, axes = plt.subplots(figsize=(6, 3.8))
    #axes.grid(alpha=0.8,axis="y")
    axes.grid(alpha=0.8,axis="y")
    width = 0.25
    labels = ['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO']
    plt.xlabel("Vegetation type",fontsize=12)
    plt.ylabel("Bowen ratio",fontsize=12)
    plt.tick_params(labelsize=10)
    x =list(range(len(labels)))
    #plt.plot([0,12], [0,0], linestyle='--', color='k',  linewidth=1)
    plt.axhline(0,linestyle='--', color='k',  linewidth=0.4)       
    error_params=dict(elinewidth=0.6,ecolor='black',capsize=1.5,alpha=0.4)
    plt.bar(x, y.b_N, width=width, label='Northern Hemisphere',yerr=y.sem_N,error_kw=error_params,fc='#4984b8',tick_label=labels,edgecolor= 'black',lw=0)
    for a,b in zip(x,y.b_N):
        if b < 0:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=5)
        else:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=5)
    
    for i in range(12):
        x[i] = x[i] - width 
    plt.bar(x, y.b, width=width, label='Global',yerr=y.sem_all,error_kw=error_params,fc='#ffb07c',edgecolor= 'black',lw=0)
    for a,b in zip(x,y.b):
        if b < 0:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=5)
        else:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=5)
    
    for i in range(12):
        x[i] = x[i] + width *2
    plt.bar(x, y.b_S, width=width, label='Southern Hemisphere',yerr=y.sem_S,error_kw=error_params, fc='mediumturquoise',edgecolor= 'black',lw=0,alpha=0.8)
    for a,b in zip(x,y.b_S):
        if b < 0:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=5)
        else:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=5)
            
    #自定义图例顺序       
    handles,labels = axes.get_legend_handles_labels()
    handles = [handles[1], handles[0], handles[2]]
    labels = [labels[1], labels[0], labels[2]]
    axes.legend(handles,labels,fontsize = 11,loc=2,frameon=False)
    plt.yticks([0,1,2,3,4])
    plt.savefig("Fluxnet/a_folder/figure/Figure 3.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

