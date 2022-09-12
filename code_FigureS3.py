
import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatur


def make_plot():
    df=pd.read_csv("Fluxnet/a_folder/各植被12个月份平均.csv")
    df_number=pd.read_csv("Fluxnet/a_folder/各点多年平均_去2站点.csv")
    typename=['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO']
    title=['Evergreen Needleleaf Forests','Evergreen Broadleaf Forests','Deciduous Needleleaf Forests','Deciduous Broadleaf Forests','Mixed Forests','Closed Shrublands','Open Shrublands','Woody Savannas','Savannas','Grasslands','Wetlands','Croplands']
    typename_S=['ENF_S','EBF_S','DBF_S','MF_S','WSA_S','SAV_S''GRA_S','WET_S']
    order=['a','b','c','d','e','f','g','h','i','j','k','l']
    
    fig, axes = plt.subplots(4,3,figsize=(10,9))
    #画出植被北半球的12个图
    for i in range(4):
        for j in range(3):
            dftemp=df[df['type'].isin([typename[i*3+j]])]
            axes[i][j].plot(dftemp['month'], dftemp['P_SUM'], linestyle='-', color='#4984b8',marker='o', markersize=2, linewidth=1.5,label = "NH")
            axes[i][j].fill_between(dftemp['month'],dftemp['P_SUM']-dftemp['P_SUM'].std(),dftemp['P_SUM']+dftemp['P_SUM'].std(),facecolor='#4984b8', alpha=0.12)
            axes[i][j].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
            axes[i][j].set_xticklabels('')
            axes[i][j].tick_params(labelsize=10,length=3)
            axes[i][j].set_title(title[i*3+j],fontsize=10,loc='center',y=0.83)
    
            
    #画出植被南半球的8个图
    def DrawS(m,n,nam):
        dftemp=df[df['type'].isin([nam])]
        axes[m][n].plot(dftemp['month'], dftemp['P_SUM'], linestyle='-', color='mediumturquoise',marker='o', markersize=2, linewidth=1.5,label = "SH")
        axes[m][n].fill_between(dftemp['month'],dftemp['P_SUM']-dftemp['P_SUM'].std(),dftemp['P_SUM']+dftemp['P_SUM'].std(),facecolor='mediumturquoise', alpha=0.12)
        #如果要把图例变成横向排列，ncol=2
        
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
        axes[i][0].set_ylabel("Precipitation ($mm$)",labelpad=-1,fontsize=12)
    for i in range(3):
        axes[3][i].set_xlabel("Month",labelpad=-0.5,fontsize=12)
        
    #调整最后一行刻度   
    for i in range(4):
        for j in range(3):
            axes[3][j].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
            axes[3][j].set_xticklabels([' ', ' ', '2 ', ' ','4',' ', '6', ' ','8', ' ', '10',' ', '12'], rotation=0)
            axes[3][j].tick_params(labelsize=10,length=3)

    axes[0][0].legend(loc=1,frameon=False,fontsize=7.8,bbox_to_anchor=(0.95,0.88))
    axes[2][0].set_yticks([4,8,12])
    
    plt.savefig("Fluxnet/a_folder/figure/Figure S3.png",dpi=600, bbox_inches='tight' )
    print("finished!")

if __name__=="__main__":
    make_plot()

