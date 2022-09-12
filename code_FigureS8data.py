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

def Deal_FigS8_data():
    df_h1= xr.open_dataset("CLM_data/ensmean_0615h1.nc")
    df_h1=df_h1.mean(dim='time')
    df_b1= xr.open_dataset("Fluxnet/a_folder/CLM_deal.nc") #为了读取H和LE
    df_b1=df_b1.mean(dim='time')
    
    #处理h1和影响因子
    var_h1=pd.DataFrame(columns=['1'])
    var_h1.insert(1,'tem',df_h1.TSA.values)#2m空气温度
    #df.insert(1,'Rn',df_h1.FSA.values)#吸收的短波辐射
    #df.insert(1,'sw_up',df_h1.FSR.values)#反射的短波辐射
    #df.insert(1,'lw',df_h1.FIRE.values)#出射的短波辐射
    var_h1.insert(1,'LAI',df_h1.TLAI.values)#LAI
    var_h1.insert(1,'lat',df_h1.pfts1d_lat.values)
    var_h1.insert(1,'lon',df_h1.pfts1d_lon.values)
    var_h1.insert(1,'H',df_b1.FSH_outlier.values)
    var_h1.insert(1,'LE',df_b1.EFLX_LH_TOT_outlier.values)
    var_h1.eval('b=H/LE',inplace=True)
    #去除outliers
    q_95 = var_h1['b'].quantile(q = 0.95)
    q_05 = var_h1['b'].quantile(q = 0.05)
    var_h1_out=var_h1[(var_h1['b']<q_95) & (var_h1['b']>q_05)]
    var_h1_out.reset_index(drop='True')
    var_h1_out.to_csv("Fluxnet/a_folder/FigS8_var_h1_out.csv")
    
    #接下来处理h0和影响因子
    #为了提取每个网格主导的pft，数据格式有误，先把数据重新封装
    df_pft= xr.open_dataset("CLM_data_h0/surfdata_360x720cru_simyr2000_c130418_pft_land.nc")
    df_pft1=xr.Dataset(
               {
                    "LANDFRAC_PFT": (["lat", "lon"], df_pft['LANDFRAC_PFT'].values),
                    "PCT_PFT": (["pft","lat", "lon"], df_pft['PCT_PFT'].values),
                },
                coords={
                    "pft":np.int32(list(range(0,17))),
                    "lat": df_pft.lsmlat.data,
                    "lon": df_pft.lsmlon.data              
                }
    )                 
    
    #提取每个网格主导的pft
    df_pftmax=df_pft1['PCT_PFT'].argmax(dim='pft')
    df_pftmax.to_netcdf("Fluxnet/a_folder/CLM_pftmax.nc")

    #提取H和LE
    #因为之前处理的/CLM_to_2d_120time.nc文件太大，所以先用cdo处理为多年平均
    #cdo timmean CLM_to_2d_120time.nc CLM_to_2d.nc
    df_b0= xr.open_dataset("Fluxnet/a_folder/CLM_to_2d.nc") #为了读取H和LE
    lat=df_b0.lat
    lon=df_b0.lon
    df_HLE=pd.DataFrame( )
    x=0
    for i in range(360):
        for j in range(720):
            H=df_b0.sel(lat=lat[i],lon=lon[j],pft=df_pftmax[i][j],method='nearest').FSH_outlier.values
            df_HLE.loc[x,'H']=float(H)
            LE=df_b0.sel(lat=lat[i],lon=lon[j],pft=df_pftmax[i][j],method='nearest').EFLX_LH_TOT_outlier.values
            df_HLE.loc[x,'LE']=float(LE)        
            x=x+1
        print(i)
        
    #计算bowen ratio
    var_h0=df_HLE.eval('b=H/LE')
    #处理h0数据
    df_h0= xr.open_dataset("CLM_data_h0/ensmean_0615h0.nc")
    var_h0.insert(2,'pre',df_h0.RAIN.values.flatten())
    var_h0.insert(2,'sw_out',df_h0.FSR.values.flatten())
    var_h0.insert(2,'sw_absorb',df_h0.FSA.values.flatten())
    var_h0.insert(2,'lw',df_h0.FIRE.values.flatten())
    var_h0.eval('albedo=sw_out/(sw_out+sw_absorb)',inplace=True)
    var_h0.eval('sw=sw_out+sw_out',inplace=True)
    var_h0.eval('pre_month=pre*2592000',inplace=True)#修改单位mm/s为mm/m
    var_h0.dropna(subset=['H'],inplace=True)
    var_h0.reset_index(drop='True')
    #h0数据去除outlier
    q_95 = var_h0['b'].quantile(q = 0.95)
    q_05 = var_h0['b'].quantile(q = 0.05)
    var_h0_out=var_h0[(var_h0['b']<q_95) & (var_h0['b']>q_05)]
    var_h0_out.reset_index(drop='True')
    var_h0_out.to_csv("Fluxnet/a_folder/FigS8_var_h0_out.csv")
    print("finished!")
    
if __name__=="__main__":
    Deal_FigS8_data()
