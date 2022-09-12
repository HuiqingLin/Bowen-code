import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_fitting(df, x_txt, y_txt, order=1,lw=1.5):
    parameter=np.polyfit(df[x_txt],df[y_txt],order)
    p = np.poly1d(parameter)
    y= parameter[0] * df[x_txt]  + parameter[1] 
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 50)
    correlation = df[x_txt].corr(df[y_txt])  #相关系数
    correlation=("%.2f" %correlation)
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
    #p_=stats.pearsonr(df[x_txt],df[y_txt]) #计算出来是2个值，第二个才是P_value
    #p_value=p_[1]
    #p_value=round(p_value,3)
    #p_value=str(p_value)
    plt.plot(xp, p(xp), '--',color='black',lw=lw)
    plt.text(7,65,"R="+c+"  "+s,color='k',fontsize=13)
    return c 

def make_plot():
    df=pd.read_csv("Fluxnet/a_folder/energy_balance_203.csv")
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
    
    plt.savefig("Fluxnet/a_folder/figure/Figure S7.png",dpi=600,bbox_inches='tight')
    print("finished!")
        
if __name__=="__main__":
    make_plot()


