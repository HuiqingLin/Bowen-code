#导入需要的库
import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
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

#计算月降水
def sum_df(df):
    SUM=df.groupby(["year","month"]).sum()
    SUM=pd.DataFrame(SUM)
    #计算各变量的月均值并加入“月降水”
    #df=df.groupby(["year","month"]).mean()
    #df=pd.DataFrame(df)
    #df.insert(4, 'P_SUM', SUM['P_F'])
    return SUM


#原始数据有出射短波辐射，计算反照率和月降水量总量
def Albedo(df):
    df=df[['year','month','H_F_MDS','LE_F_MDS','P_F','TA_F','VPD_F','LW_IN_F','SW_IN_F','SW_OUT']]#从原始数据中提取所需变量
    #去除出射短波辐射的空值，并计算反照率
    df_al=df[~df['SW_OUT'].isin([-9999])]
    df_al=df_al[(df_al['SW_OUT']<df_al['SW_IN_F']) & (df_al['SW_OUT']>=0)]
    df_al=df_al.groupby(["year","month"]).mean()
    df_al=pd.DataFrame(df_al)
    df_al.eval('a=SW_OUT/SW_IN_F',inplace=True)
    
    #插入反照率，并用处理过的出射短波辐射替换原始的出射短波辐射
    df=df.groupby(["year","month"]).mean()
    df.insert(4,'albedo',df_al['a'])
    df.drop('SW_OUT',axis=1, inplace=True)
    df.insert(4,'SW_OUT',df_al['SW_OUT'])
    return df
  
#对某一个类型植被多个站点的处理
def Deal(fod,IGBP):
    path = "../data/Fluxnet/"+fod
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
        #计算各变量月总值（为了得到月降水量）
        sumdf=sum_df(mydef)
        
        if 'SW_OUT' in mydef.columns:
            mydef=Albedo(mydef)
            mydef.insert(4, 'P_SUM', sumdf['P_F'])
            
        else:
            mydef=mydef[['year','month','H_F_MDS','LE_F_MDS','P_F','TA_F','VPD_F','LW_IN_F','SW_IN_F']]#从原始数据中提取所需变量
            mydef=mydef.groupby(["year","month"]).mean()
            mydef=pd.DataFrame(mydef)
            mydef.insert(4, 'P_SUM', sumdf['P_F'])
            mydef['albedo']=None
        
        #计算每个站点的12个月份的均值，加入站点名字和有效值
        mydef=mydef.groupby(["month"]).mean()
        mydef=pd.DataFrame(mydef)
        mydef.insert(1, 'name', name)
        mydef.insert(1, 'type',fod)
        mydef.insert(1, 'valid', l)
  
        #计算一种植被多个站点的多年平均，整合
        df_temp=mydef.mean(axis=0)
        df_temp=pd.DataFrame(df_temp)
        df_temp=df_temp.T
        df_temp.insert(1, 'P_year', mydef.sum().P_SUM)
        df_temp.insert(1, 'name', name)
        df_temp.insert(1, 'type',fod)
        df_temp.insert(1, 'IGBP',IGBP)
        df_site=pd.concat([df_site,df_temp])

        #一种植被的多个站点的12个月份，整合
        df_12=pd.concat([df_12,mydef])
    
    df_12.to_csv("../data/Fluxnet/all/"+fod+"_all.csv",encoding='utf-8') 
    return (df_site,df_12)

def Deal_vegetation():
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
    df_drop.to_csv("../data/processed_data/多点12个月份平均_去2站点.csv")

    df_mean=df_drop.groupby(["type","month"]).mean()
    df_mean.eval('b=H_F_MDS/LE_F_MDS',inplace=True) 
    df_mean.to_csv("../data/processed_data/各植被12个月份平均.csv")
    
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

    df_205_.eval('b=H_F_MDS/LE_F_MDS',inplace=True) 
    #去除有效值小于35%的行
    df_205drop=df_205_[df_205_['valid']>0.35]
    df_205drop.to_csv("../data/processed_data/各点多年平均_去2站点.csv")
    print("finished!")

if __name__=="__main__":
    Deal_vegetation()
