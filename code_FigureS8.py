import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import math
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as plt
import seaborn as sns


#回归分析拟合函数 
def plot_fitting(df, x_txt, y_txt, ax, order=1,lw=1.5,color='black'):
    parameter=np.polyfit(df[x_txt],df[y_txt],order)
    p = np.poly1d(parameter)
    y= parameter[0] * df[x_txt]  + parameter[1] 
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 20)
    correlation = df[x_txt].corr(df[y_txt])  #相关系数
    correlation=("%.2f" %correlation)#相关系数保留三位小数
    c=str(correlation)   
    p_=stats.pearsonr(df[x_txt], df[y_txt])
    p_value=p_[1]
    s=""
    if p_value <0.05 and (p_value > 0.01 or p_value == 0.01):
        s="*"
    elif p_value < 0.01 and (p_value > 0.001 or p_value == 0.001):
        s="**"
    elif p_value <0.001:
        s="***"
    ax.plot(xp, p(xp), '--',color=color,lw=lw)
    ax.text(df[x_txt].min(),(df[y_txt].max())*0.88,"R="+c+"  "+s)
    return c 

#Bowen ratio与影响因素
def make_plot():
    fig, axes = plt.subplots(2,4,figsize=(12,6)) 
    plt.subplots_adjust(hspace=0.25)
    names=['pre_month','tem','sw','lw','albedo','lat','lon','LAI']
    label=['precipitation($mm/month$)','Temperature ($℃$)','Shortwave radiation ($W/m^2$)','Longwave radiation ($W/m^2$)','Albedo','Latitude','Longitude','Leaf area index']
    var_h0_out=pd.read_csv('Fluxnet/a_folder/FigS8_var_h0_out.csv')
    var_h1_out=pd.read_csv('Fluxnet/a_folder/FigS8_var_h1_out.csv')
    df=[var_h0_out,var_h1_out,var_h0_out,var_h0_out,var_h0_out,var_h1_out,var_h1_out,var_h1_out]
    
    for i in range(2):
        for j in range(4):      
            sns.kdeplot(data=df[i*4+j],x=df[i*4+j][names[i*4+j]], y=df[i*4+j]['b'], cmap="YlGnBu", shade=True, bw=0.5,ax=axes[i][j])
            axes[i][j].set_xlabel(str(label[i*4+j]),fontsize=12,labelpad=0.1)
            axes[i][j].set_ylabel(" ",fontsize=14,labelpad=0.1)
            plot_fitting(df[i*4+j],names[i*4+j],'b',axes[i][j],order=1) 
            #axes[i][j].legend(loc=2,fontsize=10) 
    axes[0][0].set_ylabel("Bowen ratio",fontsize=14)
    axes[1][0].set_ylabel("Bowen ratio",fontsize=14)
    plt.savefig("Fluxnet/a_folder/figure/Figure S8.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

