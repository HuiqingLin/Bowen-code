import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def make_plot():
    df=pd.read_csv("../data/processed_data/energy_balance_203.csv")
    df.eval('EBC_closure=(1-otherheat/NETRAD)*100',inplace=True) 
    df=df.dropna(axis=0,subset = ["EBC_closure"])   
    df.reset_index(drop=True,inplace=True)
    
    fig=plt.figure(figsize=(5,5))
    s=['x','o','v','d','X','s','*','^','+','|','8','<']
    IGBP=['ENF','EBF','DBF','MF','CSH','OSH','WSA','SAV','GRA','CRO','WET','DNF']
    color=['darkseagreen','g','c','darkmagenta','hotpink','yellow','gray','red','chocolate','blue','lightblue','indigo']
    for i in range(12):
        plt.plot(df[df['IGBP']==IGBP[i]].b,df[df['IGBP']==IGBP[i]].EBC_closure, s[i],label=IGBP[i],color=color[i],markersize=5.5,alpha=0.7)
    plt.xlabel("Bowen ratio",fontsize=15)
    plt.ylabel("Energy balance closure(%)",fontsize=15)
    plt.legend(fontsize = 8,loc=4,borderaxespad = 0)
    
    plt.savefig("../data/processed_data/figure/Figure S7.png",dpi=600,bbox_inches='tight')
    print("finished!")
        
if __name__=="__main__":
    make_plot()


