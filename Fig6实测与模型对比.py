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

def data_deal():
    data= pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv")
    df=xr.open_dataset('Fluxnet/a_folder/to_2d_mean.nc')
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
    for i in range(len(data)):
        df1=df.sel(lat=lat[i],lon=lon[i],pft=lis[i],method='nearest').mean()
        FSH = df1.variables['FSH_outlier']
        data.loc[i,'H_M']=float(FSH)
        EFLX_LH_TOT = df1.variables['EFLX_LH_TOT_outlier']
        data.loc[i,'LE_M']=float(EFLX_LH_TOT)

    data.eval('b_M=H_M/LE_M',inplace=True)
    data.to_csv("Fluxnet/a_folder/各点多年平均_有经纬LAI_模型.csv")
    data=data.dropna(axis=0,subset = ["b_M"])   # 丢弃‘b_M'这列中有缺失值的行    
    return data

def plot_fitting(df, x_txt, y_txt,ax, text_x,text_y,s_x,s_y,order=1):
    parameter=np.polyfit(df[x_txt],df[y_txt],order)
    p = np.poly1d(parameter)
    y= parameter[0] * df[x_txt]  + parameter[1] 
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 50)
    correlation = df[x_txt].corr(df[y_txt])  #相关系数
    correlation=round(correlation,3)
    c=str(correlation)
    ax.plot(xp, p(xp), '--',color='blue',lw=1.5)
    plt.text(text_x,text_y,"R="+c,color='b',fontsize=14)
    p_=stats.pearsonr(df[x_txt], df[y_txt])
    p_value=p_[1]
    s=""
    if p_value <0.05 and (p_value > 0.01 or p_value == 0.01):
        s="*"
    elif p_value < 0.01 and (p_value > 0.001 or p_value == 0.001):
        s="**"
    elif p_value <0.001:
        s="***"
    plt.text(s_x,s_y,s,color='b',fontsize=14)
    return c



def make_plot(var,var_M,xlim,ylim,xlabel,ylabel,text_x,text_y,s_x,s_y):
    data=data_deal()  
    fig, axes = plt.subplots(figsize=(6,6)) 
    plt.subplots_adjust(hspace=None)
    s=['x','o','v','d','X','s','*','^','+','|']
    IGBP=['ENF','EBF','DBF','MF','CSH','OSH','WSA','SAV','GRA','CRO']
    color=['darkseagreen','g','c','darkmagenta','hotpink','yellow','gray','red','chocolate','blue']
    for i in range(10):
        axes.plot(data[data['IGBP']==IGBP[i]][var],data[data['IGBP']==IGBP[i]][var_M], s[i],label=IGBP[i],color=color[i],markersize=5.5)

    plot_fitting(data,var,var_M,axes,text_x,text_y,s_x,s_y)

    axes.plot((0,1),(0,1),transform=axes.transAxes,ls='--',c='k')
    axes.tick_params(labelsize=12,length=4)

    plt.legend(fontsize = 10,loc=4,borderaxespad = 0)
    plt.xlim(xlim,ylim)
    plt.ylim(xlim,ylim)
    plt.xlabel(xlabel,fontsize = 20)
    plt.ylabel(ylabel,fontsize = 20)
    plt.savefig("Fluxnet/a_folder/figure/Fig6.对比"+var+".jpg",dpi=300, bbox_inches='tight')
    print("finished!")

if __name__=="__main__":
    make_plot('b','b_M',-5,12,"Bowen ratio from FLUXNET","Bowen ratio from CLM",-4,8,-0.5,8)
    make_plot('H_F_MDS','H_M',-25,180,"Sensible heat from FLUXNET","Sensible heat from CLM",-16.5,140,24,140)
    make_plot('LE_F_MDS','LE_M',-5,200,"Latent heat from FLUXNET","Latent heat from CLM",5,160,45,160)
  

