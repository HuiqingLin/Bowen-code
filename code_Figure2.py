import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shapereader
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr

def make_plot():
    df=pd.read_csv('Fluxnet/a_folder/各点多年平均_去2站点.csv')
    fig=plt.figure(figsize=(8, 4))
    ax1=plt.axes(projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.add_feature(cfeature.LAND,color='whitesmoke',facecolor=0.95)
    ax1.coastlines(lw=0.05)
    # 标注坐标轴
    ax1.set_xticks([-120,-60,0,60,120,180], crs=ccrs.PlateCarree())
    ax1.set_xticklabels([-120,-60,0,60,120,180],fontsize=10)
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax1.set_yticklabels([-90, -60, -30, 0, 30, 60, 90],fontsize=10)
    # zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    # 添加网格线
    #ax.grid()
    ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='k', alpha=0.2, linestyle='--')
    bounds = [0,1,2,3,4,5]
    #colors =['blue','lightblue','#fe828c','tomato','orangered','red','#770001']
    #colors =['blue','lightblue','mediumseagreen','yellow','orange','red','#770001']
    colors =['lightblue','mediumseagreen','yellow','orange','red']
    cmap = ListedColormap(colors)
    cmap.set_over('#770001')
    cmap.set_under('b')
    norms = BoundaryNorm(bounds, cmap.N)
    point=plt.scatter(df['longitude'],df['latitude'],c=df['b'],s=23,alpha=0.95,cmap=cmap,norm=norms, edgecolors='black', linewidths=0.1,transform=ccrs.PlateCarree())
    cb1 = plt.colorbar(point, fraction=0.02, pad=0.04,boundaries= [-1]+[0]+[1]+[2]+[3]+[4]+[5]+[11],extend='both',ticks=[0,1,2,3,4,5],spacing='proportional',orientation='vertical',label="Bowen ratio")
    cb1.update_ticks()
    
    
    
    ax2 = plt.axes([0.158, 0.228, 0.17, 0.25])
    b=df['b']
    sns.set(style='whitegrid',)# 风格选择包括："white", "dark", "whitegrid", "darkgrid", "ticks"
    sns.distplot(df.b,  
                # 设置数据频率分布颜色
                hist=True,
                bins=None,
                kde=True,
                kde_kws={"color": "k", "lw": 0.7, "label": None,'linestyle':'--'},
                hist_kws={"histtype": "barstacked", "linewidth": 2,"alpha":0.9, "color": "#ffb07c"})
    
    ax2.tick_params(labelsize=12,length=4,pad=-1)
    
    #标注均值、中位数、标准差等
    plt.text(5, 0.32, "Mean="+str('{:.2f}'.format(b.mean())),rotation=0,fontsize=6)
    plt.text(5, 0.27, "Median="+str('{:.2f}'.format(b.median())),rotation=0,fontsize=6)
    plt.text(5, 0.22, "SD="+str('{:.2f}'.format(b.std())),rotation=0,fontsize=6)
    
    plt.xlim(-1,11)
    plt.xlabel("Bowen ratio",fontsize=7,labelpad=-0.5)
    plt.ylabel("Density",fontsize=7,labelpad=-0.5)
    plt.tick_params(labelsize=5)
    
    #plt.savefig("FLUXNET/a_folder/图/Figure 2.jpg",dpi=300,bbox_inches='tight')
    plt.savefig("FLUXNET/a_folder/figure/Figure 2.png",dpi=600,bbox_inches='tight')
    #plt.savefig("FLUXNET/a_folder/图/Figure 2.pdf",bbox_inches='tight') 
    print("finished!")

if __name__=="__main__":
    make_plot()

