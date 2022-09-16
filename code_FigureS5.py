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
from code_Figure5 import plot_fitting
from code_Figure5 import set_panel_color

def make_plot():
    df=pd.read_csv('../data/processed_data/各点多年平均_去2站点.csv')
    fig, axes = plt.subplots(3,3,figsize=(10,9)) 
    plt.subplots_adjust(hspace=0.28)
    name=['P_SUM','VPD_F','TA_F','SW_IN_F','LW_IN_F','albedo','latitude','longitude','LAI']
    label=['Precipitation ($mm/month$)','Vapor pressure deficit ($hPa$) ','Temperature ($℃$)','Shortwave radiation ($W/m^2$)','Longwave radiation ($W/m^2$)','Albedo','Latitude','Longitude','Leaf area index']
    for i in range(3):
        for j in range(3):
            a=axes[i][j].scatter( df[name[i*3+j]],df['LE_F_MDS'], c=df['TA_F'],s=10,alpha=1,cmap="RdYlBu_r")
            axes[i][j].set_xlabel(str(label[i*3+j]),fontsize=12,labelpad=0.6)
            axes[i][j].tick_params(labelsize=10,length=4)
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
                   
    #设置纵轴标题
    for z in range(3):
        axes[z][0].set_ylabel("Latent heat ($W/m^2$)",fontsize=12,labelpad=-1)
    
    #由于albedo有空值，latitude要分段拟合，故先拟合其他7个因子
    name_=['P_SUM','VPD_F','TA_F','SW_IN_F','LW_IN_F','longitude','LAI']
    axes_x=[0,0,0,1,1,2,2]
    axes_y=[0,1,2,0,1,1,2]
    for t in range(7):
        (c,s)=plot_fitting(df,name_[t],'LE_F_MDS',axes[axes_x[t]][axes_y[t]],order=1,lw=1.5,color='k',fontsize=10) 
        axes[axes_x[t]][axes_y[t]].text(df[name_[t]].min(),255,"R="+c+"  "+s,color='k',fontsize=10)   
 
    #拟合albedo
    df_a=df.dropna()     
    (c,s)=plot_fitting(df_a,'albedo','LE_F_MDS',axes[1][2],order=1,lw=1.5,color='k',fontsize=10)
    axes[1][2].text(0,255,"R="+c+"  "+s,color='k',fontsize=10)

    #纬度分为南北拟合
    df_1=df[df['latitude']>0]
    (c,s)=plot_fitting(df_1,'latitude','LE_F_MDS',axes[2][0],order=1,lw=1.5,color='k',fontsize=10)
    axes[2][0].text(2,255,"R="+c+"  "+s,color='k',fontsize=10)

    df_2=df[df['latitude']<0]
    (c,s)=plot_fitting(df_2,'latitude','LE_F_MDS',axes[2][0],order=1,lw=1.5,color='g',fontsize=10)
    axes[2][0].text(-40,255,"R="+c+"  "+s,color='k',fontsize=10)
    
    set_panel_color(axes)

    axes[2][2].text(4.15,290,'T ($°C$)',color='k',fontsize=8)
    plt.colorbar(a,fraction=0.04,pad=0.02)
    plt.savefig("../data/processed_data/figure/Figure S5.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

