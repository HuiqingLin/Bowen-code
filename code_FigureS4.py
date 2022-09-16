import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import scipy.stats as stats



def plot_fitting(df, x_txt, y_txt, order=1,lw=1.5):
    parameter=np.polyfit(df[x_txt],df[y_txt],order)
    p = np.poly1d(parameter)
    y= parameter[0] * df[x_txt]  + parameter[1] 
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 50)
    correlation = df[x_txt].corr(df[y_txt])  #相关系数
    correlation=("%.2f" %correlation)
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
    plt.plot(xp, p(xp), '--',color='black',lw=lw)
    plt.text(df[x_txt].min(),5,"R="+c+"  "+s,color='b',fontsize=10)
    return c 

def make_plot():
    df=pd.read_csv('../data/processed_data/各点多年平均_去2站点.csv')
    fig=plt.figure(figsize=(9,3.8))
    plt.subplot(121)
    plt.text(4.8,9.1,'Dry sites',fontsize=14)
    plt.text(-12,9.5,'(a)',fontsize=14)
    plt.ylabel("Bowen ratio",fontsize=14)
    plt.xlabel("Temperature($℃$)",fontsize=14,labelpad=-1)
    plt.ylim(0,10.5)
    a=plt.scatter( df[df['P_SUM']<10].TA_F,df[df['P_SUM']<10].b,c=df[df['P_SUM']<10].P_SUM,cmap="Spectral",s=38,vmin=0,vmax=70,alpha=1)
    plot_fitting(df[df['P_SUM']<10], 'TA_F', 'b', order=1,lw=1.5)
    
    plt.subplot(122)
    plt.ylim(0,10.5)
    plt.text(12,9.1,'Wet sites',fontsize=14)
    plt.text(2,9.5,'(b)',fontsize=14)
    b=plt.scatter( df[df['P_SUM']>30].TA_F,df[df['P_SUM']>30].b,c=df[df['P_SUM']>30].P_SUM,cmap="Spectral",s=40,vmin=0,vmax=70,alpha=1)
    plot_fitting(df[df['P_SUM']>30], 'TA_F', 'b', order=1,lw=1.5)
    plt.xlabel("Temperature($℃$)",fontsize=14,labelpad=-1)
    plt.colorbar(b,fraction=0.03,pad=0.04,label="Precipitation($mm/month$)")
    plt.savefig("../data/processed_data/figure/Figure S4.png",dpi=600, bbox_inches='tight' )
    print("finished!")
        
if __name__=="__main__":
    make_plot()

