
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shapereader
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import scipy.stats as stats
import xarray as xr
import netCDF4 as nc
import os





def make_plot():
    df=pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点旧.csv",encoding="gbk")
    #提取温度数据
    dfT = xr.open_dataset("Fluxnet/a_CRU/cru_ts4.05.2011.2020.tmp.dat.nc")
    t=dfT.tmp.mean(dim='time')
    t_fla=t.values.flatten()
    #提取降水数据
    dfP = xr.open_dataset("Fluxnet/a_CRU/cru_ts4.05.2011.2020.pre.dat.nc")
    p=dfP.pre.mean(dim='time')
    p_fla=p.values.flatten()
    #整合
    df_PT= pd.DataFrame() 
    df_PT.insert(0, 'precipitation',p_fla) 
    df_PT.insert(0, 'temperature',t_fla) 
    df_PT_na=df_PT.dropna()
    
    fig = plt.figure(figsize=(16,4) )
    gs = GridSpec(1, 100, figure=fig)#GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
    ax1=plt.axes(projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.stock_img()
    ax1.coastlines(alpha=0.7)
    # 标注坐标轴
    ax1.set_xticks([-120,-60,0,60,120,180], crs=ccrs.PlateCarree())
    ax1.set_xticklabels([-120,-60,0,60,120,180],fontsize=12)
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax1.set_yticklabels([-90, -60, -30, 0, 30, 60, 90],fontsize=12)
    # zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    # 添加网格线
    #ax.grid()
    ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='k', alpha=0.35, linestyle='--')
    
    ax1.plot(df[df['IGBP'].isin(['ENF'])]['longitude'],df[df['IGBP'].isin(['ENF'])]['latitude'], 'o',label='ENF(44)',color='darkseagreen',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['EBF'])]['longitude'],df[df['IGBP'].isin(['EBF'])]['latitude'], 'o',label='EBF(15)',color='g',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['DNF'])]['longitude'],df[df['IGBP'].isin(['DNF'])]['latitude'], 'o',label='DNF(1)',color='darkolivegreen',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['DBF'])]['longitude'],df[df['IGBP'].isin(['DBF'])]['latitude'], 'o',label='DBF(25)',color='c',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['MF'])]['longitude'],df[df['IGBP'].isin(['MF'])]['latitude'], 'o',label='MF(9)',color='darkmagenta',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['CSH'])]['longitude'],df[df['IGBP'].isin(['CSH'])]['latitude'], 'o',label='CSH(3)',color='hotpink',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['OSH'])]['longitude'],df[df['IGBP'].isin(['OSH'])]['latitude'], 'o',label='OSH(11)',color='yellow',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['WSA'])]['longitude'],df[df['IGBP'].isin(['WSA'])]['latitude'], 'o',label='WSA(6)',color='gray',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['SAV'])]['longitude'],df[df['IGBP'].isin(['SAV'])]['latitude'], 'o',label='SAV(9)',color='red',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['GRA'])]['longitude'],df[df['IGBP'].isin(['GRA'])]['latitude'], 'o',label='GRA(39)',color='chocolate',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['WET'])]['longitude'],df[df['IGBP'].isin(['WET'])]['latitude'], 'o',label='WET(21)',color='burlywood',markersize=4)
    ax1.plot(df[df['IGBP'].isin(['CRO'])]['longitude'],df[df['IGBP'].isin(['CRO'])]['latitude'], 'o',label='CRO(20)',color='blue',markersize=4)
    ax1.gridlines(linestyle='--')
    ax1.legend(loc='lower left',fontsize = 7.8)
    ax1.text(-190,95,'(a)',fontsize = 13)

    ax2 = fig.add_subplot(gs[0, 79:100])#gs[0, 0:3]中0选取figure的第一行，0:3选取figure第二列和第三列
    ax2.set_xlabel("Temperature ($℃$)",fontsize=12)
    ax2.set_ylabel("Precipitation  ($mm$)",fontsize=12,labelpad=-1)
    ax2.tick_params(labelsize=8)
    h=ax2.hist2d(x=df_PT_na.temperature, y=df_PT_na.precipitation, bins=50,vmin=0,vmax=1 ,cmap='gray_r',alpha=0.15)
    IGBP=['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO']
    c=['darkseagreen','g','darkolivegreen','c','darkmagenta','hotpink','yellow','gray','red','chocolate','burlywood','blue']
    for i in range(len(IGBP)):
        ax2.scatter(x=df[df['IGBP'].isin([IGBP[i]])].TA_F,y=df[df['IGBP'].isin([IGBP[i]])].P_year,
                    marker='o',color=c[i],alpha=0.8,label=IGBP[i],s=15)
    ax2.legend(fontsize = 8,loc=2,frameon=False,borderaxespad = 0.5)
    ax2.text(-39,720,'(b)',fontsize = 13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None) 
    
    #plt.savefig("Fluxnet/a_folder/图/Figure 1.pdf",bbox_inches='tight')
    plt.savefig("Fluxnet/a_folder/figure/Figure 1.png",dpi=600,bbox_inches='tight')
    print("finished!")

if __name__=="__main__":
    make_plot()


