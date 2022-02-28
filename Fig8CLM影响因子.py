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

#回归分析拟合函数 
def plot_fitting(df, x_txt, y_txt, ax, order=1,lw=1.5,color='black'):
    parameter=np.polyfit(df[x_txt],df[y_txt],order)
    p = np.poly1d(parameter)
    y= parameter[0] * df[x_txt]  + parameter[1] 
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 50)
    correlation = df[x_txt].corr(df[y_txt])  #相关系数
    correlation=round(correlation,3)#相关系数保留三位小数
    c=str(correlation)
    
    p_=stats.pearsonr(df[x_txt], df[y_txt])
    p_value=p_[1]
    s=""
    if p_value <0.05 and (p_value > 0.01 or p_value == 0.01):
        s="*"
    elif p_value < 0.01 and (p_value > 0.001 or p_value == 0.001):
        s="**"
    elif p_value <0.001:
        s="***"
    ax.plot(xp, p(xp), '--',color=color,lw=lw)
    ax.text(df[x_txt].min(),(df[y_txt].max())*0.88,"R="+c+"  "+s)
    return c 

def make_plot():
    df_h0= xr.open_dataset("CLM_data_h0/ensmean_0615h0.nc")
    df_h1= xr.open_dataset("CLM_data/ensmean_0615h1.nc")
    df_h1=df_h1.mean(dim='time')
    df_b= xr.open_dataset("Fluxnet/a_folder/CLM_deal.nc")
    df_b=df_b.mean(dim='time')
    
    #处理h1里的变量
    df=pd.DataFrame(columns=['1'])
    df.insert(1,'LAI',df_h1.TLAI.values)#LAI
    df.insert(1,'tem',df_h1.TSA.values)#2m空气温度
    df.insert(1,'Rn',df_h1.FSA.values)#吸收的短波辐射
    df.insert(1,'sw_up',df_h1.FSR.values)#反射的短波辐射
    df.insert(1,'lw',df_h1.FIRE.values)#出射的短波辐射
    df.insert(1,'H',df_b.FSH_outlier.values)
    df.insert(1,'LE',df_b.EFLX_LH_TOT_outlier.values)
    df.eval('b=H/LE',inplace=True)
    df.eval('albedo=sw_up/(sw_up+Rn)',inplace=True)
    df.insert(1,'lat',df_h1.pfts1d_lat.values)
    df.insert(1,'lon',df_h1.pfts1d_lon.values)
    q_95 = df['b'].quantile(q = 0.95)
    q_05 = df['b'].quantile(q = 0.05)
    df1=df[(df['b']<q_95) & (df['b']>q_05)]
    df1= df1.reset_index(drop=True)
    
    #处理h0里的变量
    lat=df_h0.lat
    lon=df_h0.lon
    df_b_var=pd.DataFrame(columns=['1'])
    x=0
    for i in range(len(lat)):
        for j in range(len(lon)):
            H=pd.DataFrame(df_b.where(df_b.pfts1d_lat==lat[i],df_b.pfts1d_lon==lon[j]).FSH_outlier.values)
            df_b_var.loc[x,'H']=float(H[~H.isin([0])].mean())
            LE=pd.DataFrame(df_b.where(df_b.pfts1d_lat==lat[i],df_b.pfts1d_lon==lon[j]).EFLX_LH_TOT_outlier.values)
            df_b_var.loc[x,'LE']=float(LE[~LE.isin([0])].mean())
            x=x+1
        print(i)
    
    df_b_var.to_csv("Fluxnet/a_folder/df_var.csv")

#    df_b_var=pd.read_csv("Fluxnet/a_folder/df_var.csv")

    df_b_var.eval('b=H/LE',inplace=True)
    df_b_var.insert(1,'precipitation',df_h0.RAIN.values.flatten())
    df_b_var=df_b_var[df_b_var['precipitation'].notna()]
    df_b_var= df_b_var.reset_index(drop=True)
    df_b_var.eval('pre_month=precipitation	*2592000',inplace=True)#修改单位mm/s为mm/m
    q_95 = df_b_var['b'].quantile(q = 0.95)
    q_05 = df_b_var['b'].quantile(q = 0.05)
    df2=df_b_var[(df_b_var['b']<q_95) & (df_b_var['b']>q_05)]
    df2= df2.reset_index(drop=True)
    
    #画图
    fig, axes = plt.subplots(2,4,figsize=(12,6)) 
    plt.subplots_adjust(hspace=0.28)
    names=['tem','Rn','albedo','lw','lat','lon','LAI']
    label=['Temperature ($℃$)','Net radiation ($W/m^2$)','Albedo','Longwave radiation ($W/m^2$)','Latitude','Longitude','Leaf area index']
    
    for i in range(2):
        for j in range(3):
            a=axes[i][j].scatter( df1[names[i*4+j]],df1['b'],s=12,alpha=0.2,c='lavender')
            axes[i][j].set_xlabel(str(label[i*4+j]),fontsize=14,labelpad=0.6)
            plot_fitting(df1,names[i*4+j],'b',axes[i][j],order=1) 
             
    axes[0][3].scatter( df1['lw'],df1['b'],s=12,alpha=0.2,c='lavender')
    axes[0][3].set_xlabel(str(label[3]),fontsize=14,labelpad=0.6)
    plot_fitting(df1,'lw','b',axes[0][3],order=1)
    
    axes[1][3].scatter( df2['pre_month'],df2['b'],s=12,alpha=0.2,c='lavender')
    axes[1][3].set_xlabel('Precipitation ($mm/month$)',fontsize=14,labelpad=0.6)
    plot_fitting(df2,'pre_month','b',axes[1][3],order=1)
    
    plt.savefig("Fluxnet/a_folder/figure/Fig8CLM与影响因子.jpg",dpi=300, bbox_inches='tight')
    print("finished!")

if __name__=="__main__":
    make_plot( )