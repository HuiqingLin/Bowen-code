import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur
import os

def make_plot( ):
    #读取数据
    df=pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv",encoding="gbk")
    #画频率分布图
    b=df['b']
    plt.figure(figsize=(6.2,4.5), dpi= 80)
    sns.set(style='whitegrid',)# 风格选择包括："white", "dark", "whitegrid", "darkgrid", "ticks"
    sns.distplot(df.b,  
                # 设置数据频率分布颜色
                hist=True,
                bins=None,
                kde=True,
                kde_kws={"color": "k", "lw": 1, "label": None,'linestyle':'--'},
                hist_kws={"histtype": "barstacked", "linewidth": 2,"alpha":0.8, "color": "#ffb07c"})
    
    #标注均值、中位数、标准差等
    plt.text(6, 0.32, "Mean="+str('{:.2f}'.format(b.mean())),rotation=0,fontsize=18)
    plt.text(6, 0.27, "Median="+str('{:.2f}'.format(b.median())),rotation=0,fontsize=18)
    plt.text(6, 0.22, "SD="+str('{:.2f}'.format(b.std())),rotation=0,fontsize=18)
    #设置坐标轴等
    plt.xlim(-1,11)
    plt.xlabel("Bowen ratio",fontsize=20)
    plt.ylabel("Density",fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize = 11,loc=2,frameon=False)
    plt.savefig("Fluxnet/a_folder/figure/Fig2频率分布.jpg",dpi=300, bbox_inches='tight' )
    print("finished!")
    
if __name__=="__main__":
    make_plot()
    
    
