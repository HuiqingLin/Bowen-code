import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shapereader
import matplotlib.ticker as mticker
from matplotlib import font_manager

def make_plot():
    #读取文件
    df=pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv",encoding="gbk")
    
    #画图
    fig=plt.figure(figsize=(8,4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    ax.coastlines(alpha=0.7)
    # 标注坐标轴
    ax.set_xticks([-120,-60,0,60,120,180], crs=ccrs.PlateCarree())
    ax.set_xticklabels([-120,-60,0,60,120,180],fontsize=9)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax.set_yticklabels([-90, -60, -30, 0, 30, 60, 90],fontsize=9)
    # zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # 添加网格线
    #ax.grid()
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='k', alpha=0.35, linestyle='--')
    #画散点图
    ax.plot(df[df['IGBP'].isin(['ENF'])]['longitude'],df[df['IGBP'].isin(['ENF'])]['latitude'], 'o',label='ENF(44)',color='darkseagreen',markersize=4)
    ax.plot(df[df['IGBP'].isin(['EBF'])]['longitude'],df[df['IGBP'].isin(['EBF'])]['latitude'], 'o',label='EBF(15)',color='g',markersize=4)
    ax.plot(df[df['IGBP'].isin(['DNF'])]['longitude'],df[df['IGBP'].isin(['DNF'])]['latitude'], 'o',label='DNF(1)',color='darkolivegreen',markersize=4)
    ax.plot(df[df['IGBP'].isin(['DBF'])]['longitude'],df[df['IGBP'].isin(['DBF'])]['latitude'], 'o',label='DBF(25)',color='c',markersize=4)
    ax.plot(df[df['IGBP'].isin(['MF'])]['longitude'],df[df['IGBP'].isin(['MF'])]['latitude'], 'o',label='MF(9)',color='darkmagenta',markersize=4)
    ax.plot(df[df['IGBP'].isin(['CSH'])]['longitude'],df[df['IGBP'].isin(['CSH'])]['latitude'], 'o',label='CSH(3)',color='hotpink',markersize=4)
    ax.plot(df[df['IGBP'].isin(['OSH'])]['longitude'],df[df['IGBP'].isin(['OSH'])]['latitude'], 'o',label='OSH(11)',color='yellow',markersize=4)
    ax.plot(df[df['IGBP'].isin(['WSA'])]['longitude'],df[df['IGBP'].isin(['WSA'])]['latitude'], 'o',label='WSA(6)',color='gray',markersize=4)
    ax.plot(df[df['IGBP'].isin(['SAV'])]['longitude'],df[df['IGBP'].isin(['SAV'])]['latitude'], 'o',label='SAV(9)',color='red',markersize=4)
    ax.plot(df[df['IGBP'].isin(['GRA'])]['longitude'],df[df['IGBP'].isin(['GRA'])]['latitude'], 'o',label='GRA(39)',color='chocolate',markersize=4)
    ax.plot(df[df['IGBP'].isin(['WET'])]['longitude'],df[df['IGBP'].isin(['WET'])]['latitude'], 'o',label='WET(21)',color='burlywood',markersize=4)
    ax.plot(df[df['IGBP'].isin(['CRO'])]['longitude'],df[df['IGBP'].isin(['CRO'])]['latitude'], 'o',label='CRO(20)',color='blue',markersize=4)
    #添加虚线
    ax.gridlines(linestyle='--')
    plt.legend(loc='lower left',fontsize = 7)
    plt.savefig("Fluxnet/a_folder/figure/Fig1空间图.jpg",dpi=300,bbox_inches='tight')
    print("finished!")

if __name__=="__main__":
    make_plot()
