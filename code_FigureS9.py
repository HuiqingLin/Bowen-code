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

def make_plot():
    df_site= pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv")
    df_site=df_site.eval('latitude_new=latitude+90')
    for i in range (len(df_site)):
        lat_360=math.ceil(df_site.loc[i,'latitude_new']/0.5)-1
        df_site.loc[i,'lat_360']=lat_360
        lon_720=math.ceil(df_site.loc[i,'longitude_new']/0.5)-1
        df_site.loc[i,'lon_720']=lon_720
    df_site['lat_360'] =pd.to_numeric(df_site['lat_360']).round(0).astype(int)
    df_site['lon_720'] =pd.to_numeric(df_site['lon_720']).round(0).astype(int)
    df_pftmax=xr.open_dataset("Fluxnet/a_folder/CLM_pftmax.nc")
    
    df_b0= xr.open_dataset("Fluxnet/a_folder/CLM_to_2d.nc") #为了读取H和LE
    df_h0= xr.open_dataset("CLM_data_h0/ensmean_0615h0.nc")
    df_CLM_extract=pd.DataFrame( )
    x=0
    for i in range(len(df_site)):
        pft_temp=df_pftmax.sel(lat=df_site.lat_360[i],lon=df_site.lon_720[i]).PCT_PFT.values
        H=df_b0.sel(lat=df_site.latitude[i],lon=df_site.longitude_new[i],pft=pft_temp,method='nearest').FSH_outlier.values
        df_CLM_extract.loc[x,'H']=float(H)
  
        LE=df_b0.sel(lat=df_site.latitude[i],lon=df_site.longitude_new[i],pft=pft_temp,method='nearest').EFLX_LH_TOT_outlier.values
        df_CLM_extract.loc[x,'LE']=float(LE)
            
        sw_absorb=df_h0.sel(lat=df_site.latitude[i],lon=df_site.longitude_new[i],method='nearest').FSA.values
        df_CLM_extract.loc[x,'sw_absorb']=float(sw_absorb)
            
        sw_out=df_h0.sel(lat=df_site.latitude[i],lon=df_site.longitude_new[i],method='nearest').FSR.values
        df_CLM_extract.loc[x,'sw_out']=float(sw_out)
            
        x=x+1
        print(i)
    
    df_CLM_extract1=df_CLM_extract.dropna(subset=['H'])
    df_CLM_extract1.reset_index(drop=True,inplace=True)
    df_CLM_extract1.eval('b=H/LE',inplace=True)
    df_CLM_extract1.eval('albedo=sw_out/(sw_absorb)',inplace=True)
    df_CLM_extract1=df_CLM_extract1.loc[(df_CLM_extract1['albedo']<0.8) ]
    df_CLM_extract1.reset_index(drop=True,inplace=True)
    
    
    fig, axes = plt.subplots(figsize=(6.5,4.8)) 
    parameter=np.polyfit(df_CLM_extract1.albedo,df_CLM_extract1.b,1)
    p = np.poly1d(parameter)
    y= parameter[0] * df_CLM_extract1['albedo']  + parameter[1] 
    xp = np.linspace(df_CLM_extract1.albedo.min(), df_CLM_extract1.albedo.max(), 20)
    correlation = df_CLM_extract1['albedo'].corr(df_CLM_extract1['b'])  #相关系数
    correlation=round(correlation,3)#相关系数保留三位小数
    c=str(correlation)
    plt.plot(xp, p(xp), '--',color='k',lw=1)
    p_=stats.pearsonr(df_CLM_extract1['albedo'], df_CLM_extract1['b'])
    p_value=p_[1]
    s=""
    if p_value <0.05 and (p_value > 0.01 or p_value == 0.01):
        s="*"
    elif p_value < 0.01 and (p_value > 0.001 or p_value == 0.001):
        s="**"
    elif p_value <0.001:
        s="***"
    plt.text(0.55,4,"R="+c+"  "+s,color='k',fontsize=14)
    
    plt.scatter(df_CLM_extract1.albedo,df_CLM_extract1.b, color='g',alpha=0.3,s=40)
    plt.xlabel("Albedo",fontsize = 15)
    plt.ylabel("Bowen ratio",fontsize = 15)
    plt.tick_params(labelsize=15)
    plt.xticks([0,0.2,0.3,0.4,0.5,0.6,0.7])
    plt.savefig("Fluxnet/a_folder/figure/Figure S9.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

