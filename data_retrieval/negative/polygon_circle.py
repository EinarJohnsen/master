import numpy as np
import json
import geog
import shapely.geometry

#BASED ON: https://gis.stackexchange.com/questions/268250/generating-polygon-representing-rough-100km-circle-around-latitude-longitude-poi/268277


def get_frost_polygon(lat, longtitude):

    #p = shapely.geometry.Point([9.071083834835958, 59.06658858430363])
    p = shapely.geometry.Point([lat, longtitude])

    n_points = 20
    d = 1 * 4000  # meters
    angles = np.linspace(0, 360, n_points)
    polygon = geog.propagate(p, angles, d)
    
    #print(json.dumps(shapely.geometry.mapping(shapely.geometry.Polygon(polygon))))
    
    x_fixed = ['{} {}'.format(x[0], x[1]) for x in polygon]
    li = ['{}' for x in x_fixed]
   
    res = "POLYGON(("+', '.join(li) + "))"
    res2 = res.format(*x_fixed)

    return res2


