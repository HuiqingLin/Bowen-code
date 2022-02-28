#!/usr/bin/env python
# coding: utf-8
#test23
# In[8]:


import os
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rxr
import earthpy as et
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
from osgeo import ogr
from pyproj import Proj, transform
from pyproj import CRS
from pyproj import Transformer


# In[20]:


#ogr读取shapefile数据
driver = ogr.GetDriverByName('ESRI Shapefile')
ganhan="bountry/bountry/outsites.shp"
ds = driver.Open(ganhan,0)
layer = ds.GetLayer()

print(ds[0])
print(ds[1])


# In[32]:


crs_4326 = CRS("EPSG:7024")
crs_4326


# In[13]:


import pyproj
print(pyproj.__version__)  # 2.4.1
print(pyproj.proj_version_str) # 6.2.1

proj = pyproj.Transformer.from_crs(9001, 4326, always_xy=True)

y1=ds.y.values, x1=ds.x.values
 = proj.transform(ds)
print(by)  # (-105.15027111593008, 39.72785727727918)
y1=ds.y.values, x1=ds.x.values


# In[16]:


#坐标系的转换
crs_WGS84 = CRS.from_epsg(4326)  # WGS84地理坐标系
crs_Krasovsky_1940_Albers = CRS.from_epsg(9001)  # 等积圆锥投影
def coordination_convert1(proj1):
    transformer = Transformer.from_crs(crs_Krasovsky_1940_Albers,crs_WGS84, always_xy=True)
    proj2 = transformer.transform(proj1[0],proj1[1])
    return proj2#一个二维向量
by=coordination_convert1(ds)


# In[3]:


#读取矢量数据的空间参考
spatial = layer.GetSpatialRef()

print(spatial)


# In[ ]:


#读取点矢量文件，复制特定属性值点、线、面并另存为shp文件
#创建输出文件
outfile = 'bountry/bountry/outsites.shp'

outds = driver.CreateDataSource(outfile)

#outlayer = outds.CreateLayer('outsites',spatial,geom_type = ogr.wkbPolygon)
outlayer = outds.CreateLayer('outsites',spatial,geom_type = ogr.wkbMultiPolygon)
infeature = layer.GetNextFeature()

idfielddefn = infeature.GetFieldDefnRef('id')##也可以从inlayer获取字段数，循环读取两个字段属性？

coverfielddefn = infeature.GetFieldDefnRef('NAME_RAIN')

print(coverfielddefn)
outlayer.CreateField(idfielddefn)

outlayer.CreateField(coverfielddefn)

#获取输出图层属性表信息

outfeaturedefn = outlayer.GetLayerDefn()
#获取输出图层属性表信息

outfeaturedefn = outlayer.GetLayerDefn()
#遍历要素
feature = layer.GetNextFeature()
while feature:
    name = feature.GetField('NAME_RAIN')
    if name == '干旱区' or name=='半干旱区':
        #如果符合条件，对应创建新的要素
        newfeature = ogr.Feature(outfeaturedefn)
        geom = feature.GetGeometryRef()
        newfeature.SetGeometry(geom)#添加点
        #添加点的字段值
        newfeature.SetField('NAME_RAIN',feature.GetField('NAME_RAIN'))
        newfeature.SetField('id',feature.GetField('id'))
        #添加要素到图层
        outlayer.CreateFeature(newfeature)
        newfeature.Destroy()
    feature.Destroy()
    feature = layer.GetNextFeature()
ds.Destroy()
outds.Destroy()


# In[ ]:


#矢量数据的画图
def plot_point(point,symbol='ko',**kwargs):
    x,y=point.GetX(),point.GetY()
    plt.plot(x,y,symbol,**kwargs)

def plot_line(line,symbol='g-',**kwargs):
    x,y=zip(*line.GetPoints())
    plt.plot(x,y,symbol,**kwargs)

def plot_polygon(poly,symbol='r-',**kwargs):
    for i in range(poly.GetGeometryCount()):
        subgeom=poly.GetGeometryRef(i)
        x,y=zip(*subgeom.GetPoints())
        plt.plot(x,y,symbol,**kwargs)

def plot_layer(filename,symbol,layer_index=0,**kwargs):
    ds=ogr.Open(filename)
    for row in ds.GetLayer(layer_index):
        geom=row.geometry()
        geom_type=geom.GetGeometryType()

        if geom_type==ogr.wkbPoint:
            plot_point(geom,symbol,**kwargs)
        elif geom_type==ogr.wkbMultiPoint:
            for i in range(geom.GetGeometryCount()):
                subgeom=geom.GetGeometryRef(i)
                plot_point(subgeom,symbol,**kwargs)

        elif geom_type==ogr.wkbLineString:
            plot_line(geom,symbol,**kwargs)
        elif geom_type==ogr.wkbMultiLineString:
            for i in range(geom.GetGeometryCount()):
                subgeom=geom.GetGeometryRef(i)
                plot_line(subgeom,symbol,**kwargs)

        elif geom_type == ogr.wkbPolygon:
            plot_polygon(geom,symbol,**kwargs)
        elif geom_type==ogr.wkbMultiPolygon:
            for i in range(geom.GetGeometryCount()):
                subgeom=geom.GetGeometryRef(i)
                plot_polygon(subgeom,symbol,**kwargs)

os.chdir('bountry/bountry')
#下面三个谁在上边就先显示谁，我就按照点，线，面来了
#plot_layer('quhua-nongye.shp','ko',markersize=5)
#plot_layer('quhua-nongye.shp','r-')
plot_layer('outsites.shp','k-',markersize=1)
plt.axis('equal')
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.show()


# In[ ]:


#矢量裁剪nc文件
monthly = xr.open_dataarray("precipitation/pre_2000_2002.nc")
monthly.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
monthly.rio.write_crs("epsg:4326", inplace=True)
Shape = geopandas.read_file('D:\G3P\DATA\Shapefile\Africa_SHP\Africa.shp', crs="epsg:4326")


clipped = monthly.rio.clip(Shape.geometry.apply(mapping), monthly.rio.crs, drop=False)

