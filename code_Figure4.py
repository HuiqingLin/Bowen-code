import pandas as pd
import matplotlib
#import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur

def make_plot():
    df=pd.read_csv("../data/processed_data/各植被12个月份平均.csv")
    df_number=pd.read_csv("../data/processed_data/各点多年平均_去2站点.csv")
    
    typename=['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO']
    title=['Evergreen Needleleaf Forests','Evergreen Broadleaf Forests','Deciduous Needleleaf Forests','Deciduous Broadleaf Forests','Mixed Forests','Closed Shrublands','Open Shrublands','Woody Savannas','Savannas','Grasslands','Wetlands','Croplands']
    typename_S=['ENF_S','EBF_S','DBF_S','MF_S','WSA_S','SAV_S''GRA_S','WET_S']
    text_y=[-0.85,0.4,-1.13,-0.33,-0.2,0.48,1.68,0.4,-0.6,0.25,-0.3,0.43]
    
    fig, axes = plt.subplots(4,3,figsize=(11,9))
    #画出植被北半球的12个图
    for i in range(4):
        for j in range(3):
            dftemp=df[df['type'].isin([typename[i*3+j]])]
            N=len(df_number[df_number['type'].isin([typename[i*3+j]])])
            axes[i][j].plot(dftemp['month'], dftemp['b'], linestyle='-', color='#4984b8',marker='o', markersize=2, linewidth=1.5,label = "NH")
            axes[i][j].fill_between(dftemp['month'],dftemp['b']-dftemp['b'].std(),dftemp['b']+dftemp['b'].std(),facecolor='#4984b8', alpha=0.12)
            
            axes[i][j].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
            axes[i][j].set_xticklabels('')
            axes[i][j].tick_params(labelsize=10,length=3)
            axes[i][j].set_title(title[i*3+j],fontsize=10,loc='center',y=0.83)
            
            axes[i][j].text(1,text_y[i*3+j],'N='+str(N)+' Range:'+("%.1f" %dftemp['b'].min())+'-'+("%.1f" %dftemp['b'].max()),
                            color='mediumblue',fontsize=8,alpha=0.85)        

    #画出植被南半球的8个图
    def DrawS(m,n,nam):
        dftemp=df[df['type'].isin([nam])]
        N=len(df_number[df_number['type'].isin([nam])])
        axes[m][n].plot(dftemp['month'], dftemp['b'], linestyle='-', color='mediumturquoise',marker='o', markersize=2, linewidth=1.5,label = "SH")
        axes[m][n].fill_between(dftemp['month'],dftemp['b']-dftemp['b'].std(),dftemp['b']+dftemp['b'].std(),facecolor='mediumturquoise', alpha=0.12)
        #axes[m][n].legend(loc=1,frameon=False,fontsize=7)
        #如果要把图例变成横向排列，ncol=2
        axes[m][n].text(7,text_y[m*3+n],'N='+str(N)+' Range:'+("%.1f" %dftemp['b'].min())+'-'+("%.1f" %dftemp['b'].max()),
                            color='darkcyan',fontsize=8,alpha=0.85)
        
    DrawS(0,0,'ENF_S')
    DrawS(0,1,'EBF_S')
    DrawS(1,0,'DBF_S')
    DrawS(1,1,'MF_S')
    DrawS(2,1,'WSA_S')
    DrawS(2,2,'SAV_S')
    DrawS(3,0,'GRA_S')
    DrawS(3,1,'WET_S')
    
    #设置轴标题
    for i in range(4):
        axes[i][0].set_ylabel("Bowen ratio",labelpad=-1,fontsize=12)
    
    for i in range(3):
        axes[3][i].set_xlabel("Month",labelpad=-0.5,fontsize=12)
        
    #调整最后一行刻度   
    for i in range(4):
        for j in range(3):
            axes[3][j].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
            axes[3][j].set_xticklabels([' ', ' ', '2 ', ' ','4',' ', '6', ' ','8', ' ', '10',' ', '12'], rotation=0)
            axes[3][j].tick_params(labelsize=10,length=3)
            
    #添加panel序号    
    axes[0][0].text(0.5,3.08,'(a)',fontsize=11)
    axes[0][1].text(0.5,1.23,'(b)',fontsize=11)
    axes[0][2].text(0.15,9.1,'(c)',fontsize=11)
    axes[1][0].text(0.5,4.35,'(d)',fontsize=11)
    axes[1][1].text(0.5,3.73,'(e)',fontsize=11)
    axes[1][2].text(0.15,2.01,'(f)',fontsize=11)
    axes[2][0].text(0.18,4.27,'(g)',fontsize=11)
    axes[2][1].text(0.5,5.28,'(h)',fontsize=11)
    axes[2][2].text(0.5,4.55,'(i)',fontsize=11)
    axes[3][0].text(0.2,5.49,'(j)',fontsize=11)
    axes[3][1].text(0.2,1.54,'(k)',fontsize=11)
    axes[3][2].text(0.2,1.034,'(l)',fontsize=11)
    
    axes[0][0].legend(loc=1,frameon=False,fontsize=7.8,bbox_to_anchor=(0.95,0.88))
    axes[0][0].set_yticks([-1,0,1,2,3])
    
    for j in range(3):
        axes[3][j].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
        axes[3][j].set_xticklabels([' ', ' ', '2 ', ' ','4',' ', '6', ' ','8', ' ', '10',' ', '12'], rotation=0)
        axes[3][j].tick_params(labelsize=10,length=3)
            
    plt.savefig("../data/processed_data/figure/Figure 4.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

