import xarray as xr
import netCDF4 as nc
import time
import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def make_plot():
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
    #读取处理过的Fluxnet站点数据（温度和年降水）
    df=pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv")
    
    #画图
    fig=plt.figure(figsize=(6,6))
    plt.xlabel("Temperature ($℃$)",fontsize=14)
    plt.ylabel("Precipitation  ($mm$)",fontsize=14)
    plt.tick_params(labelsize=12)
    
    h=plt.hist2d(x=df_PT_na.temperature, y=df_PT_na.precipitation, bins=50,vmin=0,vmax=1 ,cmap='gray_r',alpha=0.15)
    
    IGBP=['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO']
    c=['darkseagreen','g','darkolivegreen','c','darkmagenta','hotpink','yellow','gray','red','chocolate','burlywood','blue']
    for i in range(len(IGBP)):
        plt.scatter(x=df[df['IGBP'].isin([IGBP[i]])].TA_F,y=df[df['IGBP'].isin([IGBP[i]])].P_year,
                    marker='o',color=c[i],alpha=0.8,label=IGBP[i])
    
    plt.legend(fontsize = 10,loc=2,frameon=False,borderaxespad = 0.5)
    plt.savefig("Fluxnet/a_folder/figure/Fig1温度降水图.jpg",dpi=300, bbox_inches='tight' )
    print("finished!")
    
if __name__=="__main__":
    make_plot()

