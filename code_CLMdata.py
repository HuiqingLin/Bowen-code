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
                "lon": (["lon"], df.pfts1d_lon.data),
                "lat": (["lat"], df.pfts1d_lat.data),
            
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
    a = np.full([df.time.shape[0],17, df_ref.lat.shape[0], df_ref.lon.shape[0]], np.nan)
    for t in range(df.time.shape[0]):
        for i in range(0,17):
            temp = np.full_like(df_ref.landmask, np.nan)
            ind = df.pfts1d_itype_veg==i
            temp[df.pfts1d_jxy[ind]-1, df.pfts1d_ixy[ind]-1] = df[var][t,ind]
            a[t,i,:,:]=temp
        print(t)
    return  xr.DataArray(a, 
                     coords=[df.time, np.int32(list(range(0,17))), df_ref.lat, df_ref.lon],
                     dims=["time", "pft", "lat","lon"],name=var, attrs=df[var].attrs)

# Combine and save 2d array of different variables to a single dataframe 
def combine_dataframe(df,df_ref, var_list):
    da_all=[convert_to_2d(df,df_ref, i) for i in var_list]
    return  xr.merge(da_all)


#1d转为2d
def convert_to_2d_all():
    df0601= xr.open_dataset("CLM_data/ensmean_I2000Clm50Sp_360x720cru.clm2.h1.0006-01.nc")#读取文件，为了运用其经纬度等信息
    ds=xr.open_dataset("Fluxnet/a_folder/CLM_deal.nc")
    #ds=ds.groupby("time.month").mean()
    ds["bowen"]=ds.EFLX_LH_TOT_outlier/ds.FSH_outlier#计算Bowen ratio
    ds["pfts1d_itype_veg"]=df0601.pfts1d_itype_veg
    ds["pfts1d_jxy"]=df0601.pfts1d_jxy
    ds["pfts1d_ixy"]=df0601.pfts1d_ixy

    var_list=['EFLX_LH_TOT_outlier','FSH_outlier','bowen']
    #var_list=['EFLX_LH_TOT_outlier']
    df_HLHBowen=combine_dataframe(ds,df0601, var_list)#运行 
    df_HLBowen.to_netcdf("Fluxnet/a_folder/CLM_to_2d_120time.nc")


def compare_data_deal( ):
    #inorder to compare FLUXNET data and CLM simulation
    data= pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv")
    df=xr.open_dataset('Fluxnet/a_folder/CLM_to_2d_120time.nc')
    #模型中WET和DNF是空值，故不对这两种类型作对比
    data=data[ ~data['IGBP'].str.contains('WET')]
    data=data[ ~data['IGBP'].str.contains('DNF')]
    data.reset_index(drop=True,inplace=True)  
     
    #pft和IGBP植被类型转化
    lis=[]
    for i in range(len(data)):
        if (data.loc[[i],['IGBP']]=='ENF').bool():
            lis.append([1,2])
        elif (data.loc[[i],['IGBP']]=='EBF').bool():
            lis.append([4,5])
        elif (data.loc[[i],['IGBP']]=='DBF').bool():
            lis.append([6,7,8])
        elif (data.loc[[i],['IGBP']]=='MF').bool():
            lis.append([1,2,3,4,5,6,7,8])
        elif (data.loc[[i],['IGBP']]=='CSH').bool():
            lis.append([9,10,11])
        elif (data.loc[[i],['IGBP']]=='OSH').bool():
            lis.append([9,10,11])
        elif (data.loc[[i],['IGBP']]=='WSA').bool():
            lis.append([1,2,3,4,5,6,7,8,9,10,11])
        elif (data.loc[[i],['IGBP']]=='SAV').bool():
            lis.append([1,2,3,4,5,6,7,8,9,10,11])
        elif (data.loc[[i],['IGBP']]=='GRA').bool():
            lis.append([12,13,14])
        elif (data.loc[[i],['IGBP']]=='CRO').bool():
            lis.append([15])
              
    lat=data.latitude
    lon=data.longitude_new
    begin=data.begin
    end=data.end
    for i in range(len(data)):
        df1=df.sel(time=slice(str(begin[i]), str(end[i]))).sel(lat=lat[i],lon=lon[i],pft=lis[i],method='nearest').mean()
        FSH = df1.variables['FSH_outlier']
        data.loc[i,'H_M']=float(FSH)
        EFLX_LH_TOT = df1.variables['EFLX_LH_TOT_outlier']
        data.loc[i,'LE_M']=float(EFLX_LH_TOT)   
        print(i)
    
    data.eval('b_M=H_M/LE_M',inplace=True)
    data.to_csv("Fluxnet/a_folder/各点多年平均_有经纬LAI_模型.csv")
        
    

if __name__=="__main__":
    #time_tran_global()
    #convert_to_2d_all()
    compare_data_deal()

