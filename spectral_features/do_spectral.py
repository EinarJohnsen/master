import spectral_clustering_ as sc
import pandas as pd
import transform_epsg_coords as tec
import numpy as np

data = pd.read_csv("path_to_get_lat_long.csv", sep=";", encoding="latin-1")

data = data[["geometri"]]

datas_list = []

for element in data.values:

    try:
        points = element[0].split("(")[1].split(" ")[0:2]
        transformed_points = tec.transform_coords(points[0], points[1])
    except:
        points = element[0].split("(")[1].split(")")[0].split(" ")
        transformed_points = tec.transform_coords(points[0], points[1])

    datas_list.append([transformed_points[0], transformed_points[1]])


model = sc.spectralEmbedding()
# Number of components and list of lat-long. 
emb = model.runSpectralEmbedding(datas_list, n_components=10)

