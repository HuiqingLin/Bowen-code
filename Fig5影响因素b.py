import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
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
    ax.plot(xp, p(xp), '--',color=color,lw=lw,label="R="+c+"  "+s)
    return c 

def make_plot(var, y_name):
    #读取数据
    df=pd.read_csv('Fluxnet/a_folder/各点多年平均_去2站点.csv')
    
    #画图
    fig, axes = plt.subplots(3,3,figsize=(10,9)) 
    plt.subplots_adjust(hspace=0.28)
    
    name=['P_SUM','VPD_F','TA_F','SW_IN_F','LW_IN_F','albedo','latitude','longitude','LAI']  #不同的影响因子
    label=['Precipitation ($mm/month$)','Vapor pressure deficit ($hPa$) ','Temperature ($℃$)','Shortwave radiation ($W/m^2$)','Longwave radiation ($W/m^2$)','Albedo','Latitude','Longitude','Leaf area index']
    for i in range(3):
        for j in range(3):
            a=axes[i][j].scatter( df[name[i*3+j]],df[var], c=df['TA_F'],s=10,alpha=1,cmap="RdYlBu_r")
            axes[i][j].set_xlabel(str(label[i*3+j]),fontsize=14,labelpad=0.6)
            
            axes[i][j].legend(loc=2,fontsize=12)
            axes[i][j].tick_params(labelsize=10,length=4)
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
            
    
    for z in range(3):
        axes[z][0].set_ylabel(y_name,fontsize=14,labelpad=-1) #设置纵轴标题
    
    
    
    #由于albedo有空值，latitude要分段拟合，故先拟合其他7个因子
    name_=['P_SUM','VPD_F','TA_F','SW_IN_F','LW_IN_F','longitude','LAI']
    axes_x=[0,0,0,1,1,2,2]
    axes_y=[0,1,2,0,1,1,2]
    for t in range(7):
        plot_fitting(df,name_[t],var,axes[axes_x[t]][axes_y[t]],order=1,color='black')
        axes[axes_x[t]][axes_y[t]].legend(loc=2,fontsize=10)
        
        
    #拟合albedo
    df_a=df.dropna()
    plot_fitting(df_a,'albedo',var,axes[1][2],order=1,color='black')  
    axes[1][2].legend(loc=2,fontsize=10)
        
        
        
    #纬度分为南北半球拟合 
    df_1=df[df['latitude']>0]
    plot_fitting(df_1,'latitude',var,axes[2][0],order=1,color='black')
    axes[2][0].legend(loc=2,fontsize=10)
    df_2=df[df['latitude']<0]
    plot_fitting(df_2,'latitude',var,axes[2][0],order=1,color='green')  
    axes[2][0].legend(loc=2,fontsize=10) 
    
    
    #设置背景颜色
    #气候因子
    for i in range(2):
        for j in range(3):
            axes[i][j].patch.set_facecolor("orange")             
            axes[i][j].patch.set_alpha(0.06)      
    #地理因子
    for i in range(2):
        axes[2][i].patch.set_facecolor("gray")             
        axes[2][i].patch.set_alpha(0.08) 
    #生物因子
    axes[2][2].patch.set_facecolor("blue")             
    axes[2][2].patch.set_alpha(0.06) 
        
        
        
    #局部调整    
    axes[2][2].text(4,20.5,'T (℃)',color='k',fontsize=8)
    axes[2][0].set_xticks([-25,0,25,50,75])
    plt.colorbar(a,fraction=0.04,pad=0.02)
    plt.savefig("Fluxnet/a_folder/figure/Fig5"+var+".jpg",dpi=300,bbox_inches='tight')#保存图片
    print("finished!")

if __name__=="__main__":
    make_plot('b',"Bowen ratio")
    make_plot('H_F_MDS',"Sensible heat")
    make_plot('LE_F_MDS',"Latent heat")
