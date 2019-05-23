import pandas as pd
import transform_epsg_coords as tc
import numpy as np
import polygon_circle as pc
import requests 
import pickle
import time

temp_data = pd.read_csv("ulykker_norge.csv",sep=";", encoding="latin-1")


data = temp_data.filter(['Ulykkesdato','Ulykkestidspunkt','Ukedag', 'geometri', "Historisk vegkategori", "vegobjektid"], axis=1)
data = data[data["Historisk vegkategori"] != "Skogsbilveg"]
data = data[data["Historisk vegkategori"] != "Privat veg"]
data = data[pd.notnull(data["geometri"])]

data = data[213300:-1]

print(len(data))

# ID a8918685-1115-424e-a297-dd515a542db5

def get_period(accident_date):
    try:
        _year = int(accident_date.split("-")[0])
        _period_string = str(_year)+"-01-01/"+str(_year+1)+"-01-01"
        return _period_string
    except:
        print("Not a year????")
        return None
   

def get_time_series(sources, time):
    response = requests.get('https://frost.met.no/observations/availableTimeSeries/v0.jsonld?sources={}&referencetime={}&elements=air_temperature,grass_temperature,soil_temperature,mean(wind_speed_of_gust PT1H), sum(precipitation_amount PT1H), surface_snow_thickness, cloud_area_fraction, mean(grass_temperature PT1H), mean(surface_snow_thickness PT1H)'.format(sources, time), auth=('FROST AUTH KEY', ''))

    try:
        dat = response.json()
    except:
        print("No JSON")
    
    return response.status_code, dat

def get_frost_data(points):
    response = requests.get('https://frost.met.no/sources/v0.jsonld?geometry={}'.format(points), auth=('FROST AUTH KEY', ''))

    try:
        dat = response.json()
    except:
        print("No JSON")

    return response.status_code, dat


data_dict = {}

c1 = 0
c2 = 0
x = 1
item_count = []

for element in data.values:
    try:
        point1,point2 = element[3].split(" ")[2][1:],element[3].split(" ")[3]
    except:
        
        point1, point2 = element[3].split(" ")[1][1:], element[3].split(" ")[2][:-1]
       
    temp_res1, temp_res2 = tc.transform_coords(point1, point2)
    polygon = pc.get_frost_polygon(temp_res1, temp_res2)
    resp_code, json = get_frost_data(polygon)
    

    if x % 10000 == 0:
        print("404: ", c1, "200: ", c2, " Avg: ", sum(item_count)/(len(item_count)), " total:", x)

        output = open('available_weather.pkl', 'wb')
        pickle.dump(data_dict, output)
        output.close()

    x += 1 

    if resp_code == 404:
        c1 += 1 
        continue
    elif resp_code == 200:
        
        sensors = []
        for ele in json["data"]:
            sensors.append(ele["id"])
        
        sensor_string = ','.join(sensors)
        time_period_string = get_period(element[0])

        if time_period_string == None:
            continue

        response, time_series_json = get_time_series(sensor_string, time_period_string)
        
        if response == 404:
            continue

        try: 
            time_series_json["data"]
        except:
            print(time_series_json)
            continue
        
        c2 += 1

        data_dict[element[5]] = {"data": time_series_json["data"]}
        item_count.append(json["totalItemCount"])

    else:
        continue

    
print("400 ", c1)
print("200 ", c2)
print("Avg:", sum(item_count)/(len(item_count)))

output = open('available_weather.pkl', 'wb')
pickle.dump(data_dict, output)
output.close()
