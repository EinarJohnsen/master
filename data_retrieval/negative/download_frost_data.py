import pandas as pd 
import requests
import polygon_circle 
import transform_epsg_coords
import datetime
import pickle
import logging

logging.basicConfig(filename='download.log',level=logging.INFO)

temp_data = pd.read_csv("data_path.csv",sep=";", encoding="latin-1")


def get_period(accident_date):
    try:
        temp = accident_date.split("-")
        date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2]))
        date += datetime.timedelta(days=1)
        
        period_string = str(accident_date) + "/" + str(date).split(" ")[0]

        return period_string
    except:
        #print("Not a date")
        return None
   
def get_frost_data(points):
    response = requests.get('https://frost.met.no/sources/v0.jsonld?geometry={}'.format(points), auth=('FROST AUTH KEY', ''))

    try:
        dat = response.json()
    except:
        #print("No JSON")
        return None

    return response.status_code, dat

def download_data(sensor_data, period):

    response = requests.get("https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=air_temperature&timeresolutions=PT1H, PT30M, PT10M".format(sensor_data, period), auth=('FROST AUTH KEY', ''))
    
    if response.status_code == 404:
        #print("404 - 1")
        return None

    try:
        data = response.json()

    except:
        #print("No JSON")
        return None
    
    response_2 = requests.get("https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=sum(precipitation_amount PT1H), sum(precipitation_amount PT30M), sum(precipitation_amount PT10M)&timeresolutions=PT1H, PT10M, PT30M".format(sensor_data, period), auth=('FROST AUTH KEY', ''))
    
    if response_2.status_code == 404:
        #print("404 - 2")
        return None
    try:
        data_2 = response_2.json()
        #print("should be good")
    except:
        #print("No JSON")
        return None
    
    response_3 = requests.get("https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=surface_snow_thickness, mean(grass_temperature PT1H), grass_temperature, cloud_area_fraction&timeresolutions=PT6H, PT1H".format(sensor_data, period), auth=('FROST AUTH KEY', ''))
    
    if response_3.status_code == 404:
        return [response.status_code, response_2.status_code, None, data, data_2, None]

    try:
        data_3 = response_3.json()
    except:
        #print("No JSON")
        return [response.status_code, response_2.status_code, None, data, data_2, None]

    return [response.status_code, response_2.status_code, response_3.status_code, data, data_2, data_3]


data_dict = {}
no_data = []

counter_1 = 0 
counter_2 = 0

for c in temp_data.values:
    d = [c[-5], c[-4], c[-3], c[-2], c[-1], c[2], c[4]]
    
    period = get_period(d[-1])
    data_dict[d[-2]] = {}

    for x in range(5):

        element = d[x]
        

        if d[x] != "ukjent":
            try:
                point1,point2 = element.split(" ")[2][1:],element.split(" ")[3]

            except:
                point1, point2 = element.split(" ")[1][1:], element.split(" ")[2][:-1]

            point1, point2 = transform_epsg_coords.transform_coords(point1, point2)
            polygon = polygon_circle.get_frost_polygon(point1, point2)

            
            status_code, json_data = get_frost_data(polygon)
            if status_code == 404 or status_code == 500:
                continue
            
            sensors = [x["id"] for x in json_data["data"]]
            sensor_string = ','.join(sensors)
            
            
            res = download_data(sensor_string, period)
            if res != None:
                #print(res[0], res[1], res[2])
                
                data_dict[d[-2]]["col_"+str(x)] = {}
                data_dict[d[-2]]["col_"+str(x)]["point"] = ((point1, point2))
                data_dict[d[-2]]["col_"+str(x)]["air_temperature"] = res[3]["data"]
                data_dict[d[-2]]["col_"+str(x)]["rain"] = res[4]["data"]
                if res[2] != None: 
                    data_dict[d[-2]]["col_"+str(x)]["other"] = res[5]["data"]
                #a_row.append("Data")
                
    
                
    if data_dict[d[-2]] != {}:
        output = open('data/sensor_{}.pkl'.format(d[-2]), 'wb')
        pickle.dump(data_dict, output)
        output.close()
        counter_1 += 1 
        data_dict = {}
    else:
        counter_2 += 1 
        no_data.append(d[-2])
        output = open('no_data_jonas_2.pkl', 'wb')
        pickle.dump(no_data, output)
        output.close()
        logging.info("Sensors with data: " + str(counter_1)  + " - Total: " + str(counter_2) + " - Last accident " + str(no_data[-1]))
