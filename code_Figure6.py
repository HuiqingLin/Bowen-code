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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_fitting(df, x_txt, y_txt,ax, order=1):
    parameter=np.polyfit(df[x_txt],df[y_txt],order)
    p = np.poly1d(parameter)
    y= parameter[0] * df[x_txt]  + parameter[1] 
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 50)
    correlation = df[x_txt].corr(df[y_txt])  #相关系数
    correlation=("%.2f" %correlation)
    c=str(correlation)
    ax.plot(xp, p(xp), '--',color='blue',lw=1.5)
    p_=stats.pearsonr(df[x_txt], df[y_txt])
    p_value=p_[1]
    s=""
    if p_value <0.05 and (p_value > 0.01 or p_value == 0.01):
        s="*"
    elif p_value < 0.01 and (p_value > 0.001 or p_value == 0.001):
        s="**"
    elif p_value <0.001:
        s="***"    
    return (c,s)

def make_plot():
    #数据处理
    data= pd.read_csv("Fluxnet/a_folder/各点多年平均_有经纬LAI_模型.csv")
    data=data.dropna(axis=0,subset = ["b_M"])   # 丢弃‘b_M'这列中有缺失值的行    
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
    #排序
    order=[10,5,3,2,1,9,4,6,8,7]
    com.insert(1,'order',order)
    com.sort_values(by='order',ascending=True, inplace=True) 
    
    m=['x','o','v','d','X','s','*','^','+','|']
    IGBP=['ENF','EBF','DBF','MF','CSH','OSH','WSA','SAV','GRA','CRO']
    color=['darkseagreen','g','c','darkmagenta','hotpink','yellow','gray','red','chocolate','blue']
    
    #绘图
    fig = plt.figure(figsize=(12,12))
    gs = GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, 0:1])
    ax2 = plt.subplot(gs[1, 0:1])
    ax3 = plt.subplot(gs[2, 0:1])
    ax4 = plt.subplot(gs[0, 1:3])
    ax5 = plt.subplot(gs[1, 1:3])
    ax6 = plt.subplot(gs[2, 1:3])
    
    #figure a
    for i in range(10):
        ax1.plot(data[data['IGBP']==IGBP[i]].b,data[data['IGBP']==IGBP[i]].b_M, m[i],label=IGBP[i],color=color[i],markersize=5.5)
    (c,s)=plot_fitting(data,'b','b_M',ax1)
    ax1.plot((0,1),(0,1),transform=ax1.transAxes,ls='-',c='k',lw=0.5)
    ax1.text(-4,9.8,"R="+c+" "+s,color='b',fontsize=10)
    ax1.tick_params(labelsize=12,length=4)
    x=data.b.mean()
    y=data.b_M.mean()
    ax1.text(-4,8.8,"N=148",color='b',fontsize=10)
    ax1.text(-4,7.68,r'$\overline{x}$'+'='+("%.2f" %x),color='b',fontsize=10)
    ax1.text(-4,6.6,r'$\overline{y}$'+'='+("%.2f" %y),color='b',fontsize=10)
    ax1.legend(fontsize =6.8,loc=4,borderaxespad = 0)
    ax1.set_xlim(-5,12)
    ax1.set_ylim(-5,12)
    ax1.set_xlabel("Bowen ratio from FLUXNET",fontsize = 12,labelpad=0)
    ax1.set_ylabel("Bowen ratio from CLM",fontsize = 12)
    
    
    #figure b
    for i in range(10):
        ax2.plot(data[data['IGBP']==IGBP[i]].H_F_MDS,data[data['IGBP']==IGBP[i]].H_M, m[i],label=IGBP[i],color=color[i],markersize=5.5)
    (c,s)=plot_fitting(data,'H_F_MDS','H_M',ax2)
    ax2.plot((0,1),(0,1),transform=ax2.transAxes,ls='-',c='k',lw=0.7)
    ax2.text(-16.5,205,"R="+c+" "+s,color='b',fontsize=10)
    ax2.tick_params(labelsize=12,length=4)
    x=data.H_F_MDS.mean()
    y=data.H_M.mean()
    ax2.text(-16.5,188,"N=148",color='b',fontsize=10)
    ax2.text(-16.5,169,r'$\overline{x}$'+'='+("%.2f" %x),color='b',fontsize=10)
    ax2.text(-16.5,152,r'$\overline{y}$'+'='+("%.2f" %y),color='b',fontsize=10)
    #ax2.legend(fontsize = 6.5,loc=4,borderaxespad = 0)
    ax2.set_xlim(-25,250)
    ax2.set_ylim(-25,250)
    ax2.set_xlabel("Sensible heat from FLUXNET ($W/m^2$)",fontsize = 12,labelpad=0)
    ax2.set_ylabel("Sensible heat from CLM ($W/m^2$)",fontsize = 12)
    
    #figure c
    for i in range(10):
        ax3.plot(data[data['IGBP']==IGBP[i]].LE_F_MDS,data[data['IGBP']==IGBP[i]].LE_M, m[i],label=IGBP[i],color=color[i],markersize=5.5)
    (c,s)=plot_fitting(data,'LE_F_MDS','LE_M',ax3)
    ax3.plot((0,1),(0,1),transform=ax3.transAxes,ls='-',c='k',lw=0.7)
    ax3.text(5,173,"R="+c+" "+s,color='b',fontsize=10)
    ax3.tick_params(labelsize=12,length=4)
    
    x=data.LE_F_MDS.mean()
    y=data.LE_M.mean()
    ax3.text(5,163,"N=148",color='b',fontsize=10)
    ax3.text(5,148,r'$\overline{x}$'+'='+("%.2f" %x),color='b',fontsize=10)
    ax3.text(5,135,r'$\overline{y}$'+'='+("%.2f" %y),color='b',fontsize=10)
    ax3.set_xlim(-5,200)
    ax3.set_ylim(-5,200)
    ax3.set_xlabel("Latent heat from FLUXNET ($W/m^2$)",fontsize = 12,labelpad=0)
    ax3.set_ylabel("Latent heat from CLM ($W/m^2$)",fontsize = 12)
    
    
    #figure d
    width = 0.25
    labels = ['ENF','EBF','DBF','MF','CSH','OSH','WSA','SAV','GRA','CRO']
    ax4.set_ylabel("Bowen ratio",fontsize=14,labelpad=6)
    ax4.axhline(0,linestyle='--', color='k',  linewidth=0.4)       
    x =[1,2,3,4,5,6,7,8,9,10]
    x = np.arange(len(x))
    error_params=dict(elinewidth=0.6,ecolor='black',capsize=1.5,alpha=0.4)
    ax4.bar(x,com.b,width,label='FLUXNET',yerr=com.b_sem,error_kw=error_params,fc='#fe7b7c')
    for a,b in zip(x,com.b):
        if b < 0:
            ax4.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=9)
        else:
            ax4.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=9)
    ax4.bar(x+0.25,com.b_M,width,label='CLM',yerr=com.b_M_sem,error_kw=error_params,fc='#06c2ac',edgecolor= 'black',lw=0)
    for a,b in zip(x+0.25,com.b_M):
        if b < 0:
            ax4.text(a, b,'%.1f'%b, ha = 'center',va = 'top',fontsize=9)
        else:
            ax4.text(a, b,'%.1f'%b, ha = 'center',va = 'bottom',fontsize=9)
    ax4.tick_params(labelsize=12,length=4)
    ax4.set_xticks(x+width/2)
    ax4.set_xticklabels(['ENF\n(35)','EBF\n(10)','DBF\n(17)','MF\n(8)','CSH\n(3)',
                          'OSH\n(6)','WSA\n(6)','SAV\n(9)','GRA\n(36)','CRO\n(18)'], rotation=0,fontsize = 12)
    ax4.legend(fontsize = 10,loc=1,frameon=False,borderaxespad = 0)
    
    #figure e
    ax5.set_ylabel("Sensible heat ($W/m^2$)",fontsize=13,labelpad=2)
    ax5.axhline(0,linestyle='--', color='k',  linewidth=0.4)       
    error_params=dict(elinewidth=0.6,ecolor='black',capsize=1.5,alpha=0.4)
    ax5.bar(x,com.H_F_MDS,width,label='FLUXNET',yerr=com.H_F_MDS_sem,error_kw=error_params,fc='#fe7b7c')
    for a,b in zip(x,com.H_F_MDS):
        if b < 0:
            ax5.text(a, b,'%.0f'%b, ha = 'center',va = 'top',fontsize=9)
        else:
            ax5.text(a, b,'%.0f'%b, ha = 'center',va = 'bottom',fontsize=9)
    ax5.bar(x+0.25,com.H_M,width,label='CLM',yerr=com.H_M_sem,error_kw=error_params,fc='#06c2ac',edgecolor= 'black',lw=0)
    for a,b in zip(x+0.25,com.H_M):
        if b < 0:
            ax5.text(a, b,'%.0f'%b, ha = 'center',va = 'top',fontsize=9)
        else:
            ax5.text(a, b,'%.0f'%b, ha = 'center',va = 'bottom',fontsize=9)
    ax5.set_xticks(x+width/2)
    ax5.set_xticklabels(['ENF\n(35)','EBF\n(10)','DBF\n(17)','MF\n(8)','CSH\n(3)',
                          'OSH\n(6)','WSA\n(6)','SAV\n(9)','GRA\n(36)','CRO\n(18)'], rotation=0,fontsize = 12)
    ax5.legend(fontsize =10,loc=1,frameon=False,borderaxespad = 0)
    
    #figure f
    ax6.set_xlabel("Vegetation type",fontsize=14)
    ax6.set_ylabel("Latent heat ($W/m^2$)",fontsize=13,labelpad=2)
    ax6.axhline(0,linestyle='--', color='k',  linewidth=0.4)       
    error_params=dict(elinewidth=0.6,ecolor='black',capsize=1.5,alpha=0.4)
    ax6.bar(x,com.LE_F_MDS,width,label='FLUXNET',yerr=com.LE_F_MDS_sem,error_kw=error_params,fc='#fe7b7c')
    for a,b in zip(x,com.LE_F_MDS):
        if b < 0:
            ax6.text(a, b,'%.0f'%b, ha = 'center',va = 'top',fontsize=9)
        else:
            ax6.text(a, b,'%.0f'%b, ha = 'center',va = 'bottom',fontsize=9)
    ax6.bar(x+0.25,com.LE_M,width,label='CLM',yerr=com.LE_M_sem,error_kw=error_params,fc='#06c2ac',edgecolor= 'black',lw=0)
    for a,b in zip(x+0.25,com.LE_M):
        if b < 0:
            ax6.text(a, b,'%.0f'%b, ha = 'center',va = 'top',fontsize=9)
        else:
            ax6.text(a, b,'%.0f'%b, ha = 'center',va = 'bottom',fontsize=9)
    ax6.set_xticks(x+width/2)
    ax6.set_xticklabels(['ENF\n(35)','EBF\n(10)','DBF\n(17)','MF\n(8)','CSH\n(3)',
                          'OSH\n(6)','WSA\n(6)','SAV\n(9)','GRA\n(36)','CRO\n(18)'], rotation=0,fontsize = 12)
    ax6.legend(fontsize = 10,loc=1,frameon=False,borderaxespad = 0)
    
    ax1.text(-9.8,12,'(a)',fontsize = 15)
    ax2.text(-103,255,'(c)',fontsize = 15)
    ax3.text(-62,205,'(e)',fontsize = 15)
    ax4.text(-1.5,4.95,'(b)',fontsize = 15)
    ax5.text(-1.5,215,'(d)',fontsize = 15)
    ax6.text(-1.48,228,'(f)',fontsize = 15)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None) 
    plt.savefig("Fluxnet/a_folder/figure/figure 6.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

