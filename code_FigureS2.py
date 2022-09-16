import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur

def make_plot():
    df=pd.read_csv("../data/processed_data/各点多年平均_去2站点.csv")
    fig, axes = plt.subplots(3,3,figsize=(10,10)) 
    plt.subplots_adjust(hspace=0.3)       
    
    #画图
    type=['CRO','CSH','DBF','DNF','EBF','ENF','GRA','MF','OSH','SAV','WET','WSA']
    name=['P_SUM','VPD_F','TA_F','SW_IN_F','LW_IN_F','albedo','latitude','longitude','LAI']
    label=['Precipitation ($mm/month$)','Vapor pressure deficit ($hPa$) ','Temperature ($℃$)', 'Shortwave radiation ($W/m^2$)',
           'Longwave radiation ($W/m^2$)','Albedo','Latitude','Longitude','Leaf area index',]
    IGBP=['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO']
    color=['darkseagreen','g','darkolivegreen','c','darkmagenta','hotpink','yellow','gray','red','chocolate','darkorange','blue']
    style=['x','o','P','v','d','X','s','*','^','+', '1','|']
    marksize=[8,6,6,6,6,6,6,8,6,8,8,8]
    for i in range(3):
        for j in range(3):
            for z in range(12):
                axes[i][j].plot(df[(df['IGBP']==IGBP[z]) & (df['latitude']<0)][name[i*3+j]].mean(),
                                df[(df['IGBP']==IGBP[z])& (df['latitude']<0)]['b'].mean(), style[z],
                                label=type[z],color='k',markersize=marksize[z],alpha=0.9)
                axes[i][j].plot(df[(df['IGBP']==IGBP[z]) & (df['latitude']>0)][name[i*3+j]].mean(),
                                df[(df['IGBP']==IGBP[z])& (df['latitude']>0)]['b'].mean(), style[z],
                                label=type[z],color=color[z],markersize=marksize[z],alpha=0.8)
                
            axes[i][j].tick_params(labelsize=10,length=4)
            axes[i][j].set_xlabel(str(label[i*3+j]),fontsize=14,labelpad=0.6)
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
            
    for z in range(12):
        axes[1][2].plot(df[(df['IGBP']==IGBP[z]) & (df['latitude']<0)]['albedo'].mean(),
                        df[(df['IGBP']==IGBP[z])& (df['latitude']<0)]['b'].mean(), 
                        style[z],label=type[z],color='k',markersize=marksize[z],alpha=0.9)
        axes[1][2].plot(df[(df['IGBP']==IGBP[z]) & (df['latitude']>0)]['albedo'].mean(),
                        df[(df['IGBP']==IGBP[z])& (df['latitude']>0)]['b'].mean(), 
                        style[z],label=type[z],color=color[z],markersize=marksize[z],alpha=0.8)
    
    for z in range(12):
        axes[2][1].plot(df[(df['IGBP']==IGBP[z]) & (df['latitude']<0)]['latitude'].mean(),
                        df[(df['IGBP']==IGBP[z])& (df['latitude']<0)]['b'].mean(), 
                        style[z],label=type[z],color='k',markersize=marksize[z],alpha=0.9)
        axes[2][1].plot(df[(df['IGBP']==IGBP[z]) & (df['latitude']>0)]['latitude'].mean(),
                        df[(df['IGBP']==IGBP[z])& (df['latitude']>0)]['b'].mean(), 
                        style[z],label=type[z],color=color[z],markersize=marksize[z],alpha=0.8)     
        
    for i in range (3):
        axes[i][0].set_ylabel("Bowen ratio",fontsize=14,labelpad=-1)
        
    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
    #plt.legend(loc=4.01,fontsize = 7)
    #plt.colorbar(a,fraction=0.03,pad=0.02)
    plt.savefig("../data/processed_data/figure/Figure S2.png",dpi=600,bbox_inches='tight')
    print("finished!")

if __name__=="__main__":
    make_plot()

