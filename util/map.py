import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
import transform_epsg_coords as tec

data = pd.read_csv("path_to_dataset.csv",sep=";", encoding="latin-1")
data = data[["geometri", "sample_type"]]


lats = []
longs = []
lats2 = []
longs2 = []
counter = 0 
for y in data.values:
    #print(y)
    x = [y[0]]
    
    try:
        p1,p2 = x[0].split("(")[1].split(")"[0])[0].split(" ")[0], x[0].split("(")[1].split(")"[0])[0].split(" ")[1]
    except:
        print("except")
    
    p1, p2 = tec.transform_coords(p1, p2)
    
    if y[1] == "positive":
        lats2.append(p2)
        longs2.append(p1)
    else:
        lats.append(p2)
        longs.append(p1)


print(len(lats), len(lats2))


# CODE BASED ON: https://stackoverflow.com/questions/51621615/which-geopandas-datasets-maps-are-available


df = pd.DataFrame({
                   'latitude': lats,
                   'longitude': longs})
df2 = pd.DataFrame({
                   'latitude': lats2,
                   'longitude': longs2})


gdf = gpd.GeoDataFrame(df.drop(['latitude', 'longitude'], axis=1),
                       crs={'init': 'epsg:4326'},
                       geometry=[shapely.geometry.Point(xy)
                                 for xy in zip(df.longitude, df.latitude)])

gdf2 = gpd.GeoDataFrame(df2.drop(['latitude', 'longitude'], axis=1),
                       crs={'init': 'epsg:4326'},
                       geometry=[shapely.geometry.Point(xy)
                                 for xy in zip(df2.longitude, df2.latitude)])

# Map available at GADM.org
norway = gpd.read_file("NOR_Map/")
base = norway.plot(color='white', edgecolor='black')

gdf.plot(ax=base, marker='o', color='red', markersize=4)
gdf2.plot(ax=base, marker='o', alpha=0.2, color="blue", markersize=4)


plt.show()
