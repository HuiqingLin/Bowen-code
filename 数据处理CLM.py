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

#转换UTC time为Local time
def time_tran(longitude,begintime=9,endtime=16):
    lon=longitude
    dif=round(lon/15)
    time=np.arange(0,24)
    time_local=time+dif
    time_local[time_local>=24]=time_local[time_local>=24]-24
    return (time_local>=9)& (time_local<=16)

#制作全球时间转化的掩膜文件mask
def mask_time(df):
    mask=np.zeros([24,477279])
    for x in range(477279):
        longitude=float(df.pfts1d_lon[x])
        time=time_tran(longitude)
        mask[:,x]=time
    return mask

#对所有文件做时间转化，变量只有感热潜热
def time_tran_global( ):
    df= xr.open_dataset("CLM_data/ensmean_I2000Clm50Sp_360x720cru.clm2.h1.0006-01.nc")#以该数据为例制作掩膜
    #mask=mask_time(df)
    #np.save('mask',mask)#将掩膜文件储存起来  
    mask=np.load('mask.npy') #加载掩膜文件
    years=['06','07','08','09','10','11','12','13','14','15']
    months=['01','02','03','04','05','06','07','08','09','10','11','12']
    a=np.full([120,9,477279], np.nan)
    global i
    i=0
    for year in years:
        for month in months:
            df= xr.open_dataset("CLM_data/ensmean_I2000Clm50Sp_360x720cru.clm2.h1.00"+year+"-"+month+".nc")
            variable=['EFLX_LH_TOT','FSH']
            for j in range(len(variable)):
                v=df[variable[j]].where(mask).mean(dim='time').values
                a[i, j, : ]=v
            i=i+1
            print(i)
        
    #封装为xarray
    ds = xr.Dataset(
           {
                "EFLX_LH_TOT": (["time", "pft"], a[:,0,:]),
                "FSH": (["time", "pft"], a[:,1,:]),
            },
            coords={
                "pfts1d_lon": (["pft"], df.pfts1d_lon.data),
                "pfts1d_lat": (["pft"], df.pfts1d_lat.data),
            
                "pfts1d_itype_veg": (["pft"], df.pfts1d_itype_veg.data),
            
                 "time":  pd.date_range("2006-01-01", periods=120,freq="M")
            }
        )                 

    #将封装好的xarray储存
    ds.to_netcdf("Fluxnet/a_folder/CLM_day.nc")
    
    #根据5和95百分位去除异常值，并储存
    q95_LH=ds.EFLX_LH_TOT.quantile(0.95,dim='time')
    q05_LH=ds.EFLX_LH_TOT.quantile(0.05,dim='time')
    q95_H=ds.FSH.quantile(0.95,dim='time')
    q05_H=ds.FSH.quantile(0.05,dim='time')
    EFLX_LH_TOT_outlier=ds.EFLX_LH_TOT.where((ds.EFLX_LH_TOT<q95_LH) & (ds.EFLX_LH_TOT>q05_LH))
    FSH_outlier=ds.FSH.where((ds.FSH<q95_H) & (ds.FSH>q05_H))
    ds["EFLX_LH_TOT_outlier"]=EFLX_LH_TOT_outlier
    ds["FSH_outlier"]=FSH_outlier
    ds.to_netcdf("Fluxnet/a_folder/CLM_deal.nc")
    
    
# Convert and save the 1d subgrid output of CLM to 2d data array
def convert_to_2d(df,df_ref, var):
    a = np.full([df.month.shape[0],17, df_ref.lat.shape[0], df_ref.lon.shape[0]], np.nan)
    for t in range(df.month.shape[0]):
        for i in range(0,17):
            temp = np.full_like(df_ref.landmask, np.nan)
            ind = df.pfts1d_itype_veg==i
            temp[df.pfts1d_jxy[ind]-1, df.pfts1d_ixy[ind]-1] = df[var][t,ind]
            a[t,i,:,:]=temp
        print(t)
    return  xr.DataArray(a, 
                     coords=[df.month, np.int32(list(range(0,17))), df_ref.lat, df_ref.lon],
                     dims=["month", "pft", "lat","lon"],name=var, attrs=df[var].attrs)

# Combine and save 2d array of different variables to a single dataframe 
def combine_dataframe(df,df_ref, var_list):
    da_all=[convert_to_2d(df,df_ref, i) for i in var_list]
    return  xr.merge(da_all)


#1d转为2d
def convert_to_2d_all():
    df0601= xr.open_dataset("CLM_data/ensmean_I2000Clm50Sp_360x720cru.clm2.h1.0006-01.nc")#读取文件，为了运用其经纬度等信息
    ds=xr.open_dataset("Fluxnet/a_folder/CLM_deal.nc")
    ds=ds.groupby("time.month").mean()
    ds["bowen"]=ds.EFLX_LH_TOT_outlier/ds.FSH_outlier#计算Bowen ratio
    ds["pfts1d_itype_veg"]=df0601.pfts1d_itype_veg
    ds["pfts1d_jxy"]=df0601.pfts1d_jxy
    ds["pfts1d_ixy"]=df0601.pfts1d_ixy

    var_list=['EFLX_LH_TOT_outlier','FSH_outlier','bowen']
    df_HLHBowen=combine_dataframe(ds,df0601, var_list)#运行

    df_mean=df_HLHBowen.mean(dim='month')
    df_mean.to_netcdf("Fluxnet/a_folder/to_2d_mean.nc")

if __name__=="__main__":
    time_tran_global()
    convert_to_2d_all()



