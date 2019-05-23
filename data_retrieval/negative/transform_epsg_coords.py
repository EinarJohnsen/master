from pyproj import Proj, transform


def transform_coords(point1, point2):

    inProj = Proj(init='epsg:32633')
    outProj = Proj(init='epsg:4326')
    x1,y1 = point1, point2
    x2,y2 = transform(inProj,outProj,x1,y1)
    return x2,y2

