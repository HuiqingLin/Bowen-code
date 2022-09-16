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
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as plt
import seaborn as sns 
from code_Figure5 import plot_fitting

#Bowen ratio与影响因素
def make_plot():
    fig, axes = plt.subplots(2,4,figsize=(12,6)) 
    plt.subplots_adjust(hspace=0.25)
    names=['pre_month','tem','sw','lw','albedo','lat','lon','LAI']
    label=['precipitation($mm/month$)','Temperature ($°C$)','Shortwave radiation ($W/m^2$)','Longwave radiation ($W/m^2$)','Albedo','Latitude','Longitude','Leaf area index']
    var_h0_out=pd.read_csv('../data/processed_data/FigS8_var_h0_out.csv')
    var_h1_out=pd.read_csv('../data/processed_data/FigS8_var_h1_out.csv')
    df=[var_h0_out,var_h1_out,var_h0_out,var_h0_out,var_h0_out,var_h1_out,var_h1_out,var_h1_out]
    
    for i in range(2):
        for j in range(4):      
            sns.kdeplot(data=df[i*4+j],x=df[i*4+j][names[i*4+j]], y=df[i*4+j]['b'], cmap="YlGnBu", shade=True, bw=0.5,ax=axes[i][j])
            axes[i][j].set_xlabel(str(label[i*4+j]),fontsize=12,labelpad=0.1)
            axes[i][j].set_ylabel(" ",fontsize=14,labelpad=0.1)

            (c,s)=plot_fitting(df[i*4+j],names[i*4+j],'b',axes[i][j],order=1,lw=1.5,color='k',fontsize=10)
            x=df[i*4+j][names[i*4+j]].min()
            y=df[i*4+j]['b'].max()
            axes[i][j].text(x,y,"R="+c+"  "+s,color='k',fontsize=10)
 
    axes[0][0].set_ylabel("Bowen ratio",fontsize=14)
    axes[1][0].set_ylabel("Bowen ratio",fontsize=14)
    plt.savefig("../data/processed_data/figure/Figure S8.png",dpi=600,bbox_inches='tight')
    print("finished!")
    
if __name__=="__main__":
    make_plot()

