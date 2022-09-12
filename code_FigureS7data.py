import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import scipy.stats as stats
import os

#提取时间并按列添加进表
def InsertTime(df):
    list=[]
    for i in range( len(df) ):
        list.append(str(df['TIMESTAMP_START'][i]))
    df.insert(1, 'time', list)  
    df['time'] = pd.to_datetime(df['time'])
    year1=[i.year for i in df['time']]
    month1=[i.month for i in df['time']]
    day1=[i.dayofyear for i in df['time']]
    hour1=[i.hour for i in df['time']]
    df.insert(3, 'year', year1)  
    df.insert(4, 'month', month1)  
    df.insert(5, 'dayofyear', day1)
    df.insert(6, 'hour',hour1)
    return df

#全年提取白天9-16时
def Sundf(df):
    df=df[(df['hour']>=9) & (df['hour']<=16)]
    return df
    
#去除感热和潜热为空值的行
def DelZero(df):
    df=df[~df['H_F_MDS'].isin([-9999])]
    df=df[~df['LE_F_MDS'].isin([-9999])]
    return df

#按照5%和95%的百分位去除异常
def OutliersDeal(df,cols):
    df_all=pd.DataFrame()
    u_95 = df[cols].quantile(q = 0.95)
    u_05 = df[cols].quantile(q = 0.05)
    df=df[(df[cols]<u_95) & (df[cols]>u_05)]
    df_all=pd.concat([df_all,df])
    return df_all

#对某一个类型植被多个站点的处理
def Deal(fod,IGBP):
    path = "Fluxnet/"+fod
    names=[]
    df_12=pd.DataFrame()
    global df_site
    df_site=pd.DataFrame()
    #从路径中读取文件名称
    for file in os.listdir(path):
        file_path = file
        fname,ext = os.path.splitext(file_path)
        names.append(fname)
    #根据文件名称读取文件
    for name in names:
        mydef0=pd.read_csv(path+"/"+name+".csv",encoding='utf8')
        #提取时间（年月日）
        mydef0=InsertTime(mydef0)
        #提取白天
        mydef0=Sundf(mydef0)
        #去感热和潜热的空值
        mydef=DelZero(mydef0)
        #去异常值
        mydef=OutliersDeal(mydef,'H_F_MDS')
        mydef=OutliersDeal(mydef,'LE_F_MDS')
        #计算有效值
        l=len(mydef)/len(mydef0)

        if 'G_F_MDS' in mydef.columns:
            df_var=mydef[['year','month','H_F_MDS','LE_F_MDS','G_F_MDS']]
        else:
            df_var=mydef[['year','month','H_F_MDS','LE_F_MDS']]
            df_var['G_F_MDS']=None
        
        if 'NETRAD' in mydef.columns:
            df_var['NETRAD']=mydef['NETRAD']
        else:
            df_var['NETRAD']=None
           
        df_var=df_var[~df_var['G_F_MDS'].isin([-9999])]
        df_var=df_var[~df_var['NETRAD'].isin([-9999])]
        
        #计算每个站点的12个月份的均值，加入站点名字和有效值
        mydef1=df_var.groupby(["month"]).mean()
        mydef1=pd.DataFrame(mydef1)
        mydef1.insert(1, 'name', name)
        mydef1.insert(1, 'type',fod)
        mydef1.insert(1, 'valid', l)
        df_12=pd.concat([df_12,mydef1])#一种植被的多个站点的12个月份，整合

        mydef2=df_var.mean()
        mydef2=pd.DataFrame(mydef2)
        mydef2=mydef2.T
        mydef2.insert(1, 'name', name)
        mydef2.insert(1, 'type',fod)
        mydef2.insert(1, 'IGBP',IGBP)
        mydef2.insert(1, 'valid', l)
        df_site=pd.concat([df_site,mydef2]) 
    return (df_site,df_12)

def calculate_EBC():
    #对12种植被类型，205个站点做处理，得到各站点多年平均值
    global df_205
    df_205=pd.DataFrame()
    global df_205_12
    df_205_12=pd.DataFrame()
    fod=["CRO","CSH","DBF","DBF_S","DNF","EBF","EBF_S","ENF","ENF_S","GRA","GRA_S","MF","MF_S","OSH",
        "SAV", "SAV_S","WET","WET_S","WSA","WSA_S"]
    IGBP=["CRO","CSH","DBF","DBF","DNF","EBF","EBF","ENF","ENF","GRA","GRA","MF","MF","OSH",
        "SAV", "SAV","WET","WET","WSA","WSA"]   
    for i in range(len(fod)):
        (df_site,df_12)=Deal(fod[i],IGBP[i])
        df_205=pd.concat([df_205,df_site])
        df_205_12=pd.concat([df_205_12,df_12])
        print(i)
    #去除有效值小于35%的行
    df_drop=df_205_12[df_205_12['valid']>0.35]
    df_mean=df_drop.groupby(["type","month"]).mean()
    df_mean.eval('b=H_F_MDS/LE_F_MDS',inplace=True) 
    #加入其他信息
    df_205_=df_205.sort_values(by='name')
    df_205_=df_205_.reset_index()
    df_info=pd.read_csv("Fluxnet/a_folder/经纬LAI.csv")
    df_205_.insert(len(df_205_.columns),'latitude',df_info.latitude)
    df_205_.insert(len(df_205_.columns),'longitude',df_info.longitude)
    df_205_.insert(len(df_205_.columns),'longitude_new',df_info.longitude_new)
    df_205_.insert(len(df_205_.columns),'LAI',df_info.LAI)
    df_205_.insert(len(df_205_.columns),'begin',df_info.begin)
    df_205_.insert(len(df_205_.columns),'end',df_info.end)
    df_205_[['H_F_MDS','LE_F_MDS','G_F_MDS','NETRAD']] = df_205_[['H_F_MDS','LE_F_MDS','G_F_MDS','NETRAD']].apply(pd.to_numeric)
    df_205_.eval('b=H_F_MDS/LE_F_MDS',inplace=True) 
    #去除有效值小于35%的行
    df_205drop=df_205_[df_205_['valid']>0.35]
    df_205drop.eval('otherheat=NETRAD-G_F_MDS-LE_F_MDS-H_F_MDS',inplace=True) 
    df_205drop.eval('EBC_percentage=(otherheat/NETRAD)*100',inplace=True) 
    df_205drop.to_csv("Fluxnet/a_folder/energy_balance_203.csv")
    print("finished!")
        
if __name__=="__main__":
    calculate_EBC()

