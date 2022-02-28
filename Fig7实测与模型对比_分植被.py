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

    
def com_deal():
    data=pd.read_csv("Fluxnet/a_folder/各点多年平均_有经纬LAI_模型.csv") 
    compare=data.groupby('IGBP').mean()
    compare_sem=data.groupby('IGBP').sem()
    com=pd.DataFrame(columns=['1'])
    com.insert(1,'H_M',compare.H_M)
    com.insert(1,'LE_M',compare.LE_M)
    com.insert(1,'b_M',compare.b_M)
    com.insert(1,'H_F_MDS',compare.H_F_MDS)
    com.insert(1,'LE_F_MDS',compare.LE_F_MDS)
    com.insert(1,'b',compare.b)
    com.insert(1,'H_M_sem',compare_sem.H_M)
    com.insert(1,'LE_M_sem',compare_sem.LE_M)
    com.insert(1,'b_M_sem',compare_sem.b_M)
    com.insert(1,'H_F_MDS_sem',compare_sem.H_F_MDS)
    com.insert(1,'LE_F_MDS_sem',compare_sem.LE_F_MDS)
    com.insert(1,'b_sem',compare_sem.b)
    order=[10,5,3,2,1,9,4,6,8,7]
    com.insert(1,'order',order)
    com.sort_values(by='order',ascending=True, inplace=True) 
    
    return com
    
    
#绘图
def make_plot(y_label,var,var_sem,var_M,var_M_sem):
    com=com_deal()
    
    fig, axes = plt.subplots(figsize=(8, 4))
    width = 0.25
    labels = ['ENF','EBF','DBF','MF','CSH','OSH','WSA','SAV','GRA','CRO']
    plt.xlabel("Vegetation type",fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.axhline(0,linestyle='--', color='k',  linewidth=0.4)       
    x =[1,2,3,4,5,6,7,8,9,10]
    x = np.arange(len(x))
    error_params=dict(elinewidth=0.6,ecolor='black',capsize=1.5,alpha=0.4)
    axes.bar(x,com[var],width,label='FLUXNET',yerr=com[var_sem],error_kw=error_params,fc='#fe7b7c')
    for a,b in zip(x,com[var]):
        if b < 0:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=6)
        else:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=6)

    axes.bar(x+0.25,com[var_M],width,label='CLM',yerr=com[var_M_sem],error_kw=error_params,fc='#06c2ac',edgecolor= 'black',lw=0)
    for a,b in zip(x+0.25,com[var_M]):
        if b < 0:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=6)
        else:
            plt.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=6)

    axes.set_xticks(x+width/2)
    axes.set_xticklabels(['ENF','EBF','DBF','MF','CSH','OSH','WSA','SAV','GRA','CRO'], rotation=0)
    plt.legend(fontsize = 8,loc=1,frameon=False,borderaxespad = 0)
    plt.savefig("Fluxnet/a_folder/figure/Fig7"+var+".jpg",dpi=300, bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot("Bowen ratio",'b','b_sem','b_M','b_M_sem')
    make_plot("Sensible heat($W/m^2$)",'H_F_MDS','H_F_MDS_sem','H_M','H_M_sem')
    make_plot("Latent heat($W/m^2$)",'LE_F_MDS','LE_F_MDS_sem','LE_M','LE_M_sem')
