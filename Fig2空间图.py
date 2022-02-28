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
    ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND,color='whitesmoke',facecolor=0.95)
    ax.coastlines(lw=0.05)
    
    # 标注坐标轴
    ax.set_xticks([-120,-60,0,60,120,180], crs=ccrs.PlateCarree())
    ax.set_xticklabels([-120,-60,0,60,120,180],fontsize=12)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax.set_yticklabels([-90, -60, -30, 0, 30, 60, 90],fontsize=12)
    # zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # 添加网格线
    #ax.grid()
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='k', alpha=0.2, linestyle='--')
    
    bounds = [-1,0,1,2,3,4,5,11]
    colors =['blue','lightblue','#fe828c','tomato','orangered','red','#770001']
    cmap = ListedColormap(colors)
    norms = BoundaryNorm(bounds, cmap.N)
    point=plt.scatter(df['longitude'],df['latitude'],c=df['b'],s=20,alpha=1,cmap=cmap,norm=norms, edgecolors='black', linewidths=0.1,transform=ccrs.PlateCarree())
    cb1 = plt.colorbar(point, fraction=0.02, pad=0.02,boundaries= [-1]+[0]+[1]+[2]+[3]+[4]+[5]+[11], extend='both',ticks=[0,1,2,3,4,5],spacing='proportional',orientation='vertical')
    cb1.update_ticks()
    
    plt.savefig("Fluxnet/a_folder/figure/Fig2空间图.jpg",dpi=300, bbox_inches='tight')
    print("finished!")

if __name__=="__main__":
    make_plot()
