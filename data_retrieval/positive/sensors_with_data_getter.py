import pandas as pd
import pickle 
import math

temp_data = pd.read_csv("ulykker_norge.csv",sep=";", encoding="latin-1")

data = temp_data.filter(['Ulykkesdato','Ulykkestidspunkt','Ukedag', 'geometri', "Historisk vegkategori", "vegobjektid"], axis=1)
data = data[data["Historisk vegkategori"] != "Skogsbilveg"]
data = data[data["Historisk vegkategori"] != "Privat veg"]
data = data[pd.notnull(data["geometri"])]


pkl_file = open('../../../available_weather_2.pkl', 'rb')
available_weather = pickle.load(pkl_file)
pkl_file.close()


def get_period(accident_date):
    try:
        _year = int(accident_date.split("-")[0])
        _period_string = str(_year)+"-01-01/"+str(_year+1)+"-01-01"
        return _period_string
    except:
        print("Not a year????")
        return None
   
sensor_data = {}

c = 0 
types = []
for element in available_weather:
    for x in available_weather[element]["data"]:
        types.append(x["elementId"])


for element in available_weather:
    sensor_data[element] = []
    for x in available_weather[element]["data"]:
        if x["elementId"] == "air_temperature" and (x["timeResolution"] == "PT1H" or x["timeResolution"] == "PT30M" or x["timeResolution"] == "PT10M"):
                
                try: 
                        if x["validTo"]:
                                a = x["validTo"].split("-")[0]
                                b = data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1].split("-")[0]

                                if int(a) > int(b):
                                        sensor_data[element].append("air_temperature")
                                        sensor_data[element].append(x["timeResolution"])
                                        sensor_data[element].append(x["sourceId"])


                except:
                        sensor_data[element].append("air_temperature")
                        sensor_data[element].append(x["timeResolution"])
                        sensor_data[element].append(x["sourceId"])

                        
                
                
        if x["elementId"] == "surface_snow_thickness" and (x["timeResolution"] == "PT1H" or x["timeResolution"] == "PT30M" or x["timeResolution"] == "PT10M" or x["timeResolution"] == "P1D"):

                try: 
                        if x["validTo"]:
                                a = x["validTo"].split("-")[0]
                                b = data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1].split("-")[0]

                                if int(a) > int(b):
                                        sensor_data[element].append("surface_snow_thickness")
                                        sensor_data[element].append(x["timeResolution"])
                                        sensor_data[element].append(x["sourceId"])


                except:
                        sensor_data[element].append("surface_snow_thickness")
                        sensor_data[element].append(x["timeResolution"])
                        sensor_data[element].append(x["sourceId"])

                
                
        if x["elementId"] == "grass_temperature" and (x["timeResolution"] == "PT1H" or x["timeResolution"] == "PT30M" or x["timeResolution"] == "PT6H"):
                try: 
                        if x["validTo"]:
                                a = x["validTo"].split("-")[0]
                                b = data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1].split("-")[0]

                                if int(a) > int(b):
                                       sensor_data[element].append("grass_temperature")
                                       sensor_data[element].append(x["timeResolution"])
                                       sensor_data[element].append(x["sourceId"])


                except:
                        sensor_data[element].append("grass_temperature")
                        sensor_data[element].append(x["timeResolution"])
                        sensor_data[element].append(x["sourceId"])


               
        if x["elementId"] == "sum(precipitation_amount PT1H)" and (x["timeResolution"] == "PT1H" or x["timeResolution"] == "PT30M" or x["timeResolution"] == "PT10M"):
                
                try: 
                        if x["validTo"]:
                                a = x["validTo"].split("-")[0]
                                b = data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1].split("-")[0]

                                if int(a) > int(b):
                                       sensor_data[element].append("sum(precipitation_amount PT1H)")
                                       sensor_data[element].append(x["timeResolution"])
                                       sensor_data[element].append(x["sourceId"])


                except:
                        sensor_data[element].append("sum(precipitation_amount PT1H)")
                        sensor_data[element].append(x["timeResolution"])
                        sensor_data[element].append(x["sourceId"])

                
        if x["elementId"] == "mean(grass_temperature PT1H)" and (x["timeResolution"] == "PT1H" or x["timeResolution"] == "PT30M" or x["timeResolution"] == "PT10M"):
                
                try: 
                        if x["validTo"]:
                                a = x["validTo"].split("-")[0]
                                b = data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1].split("-")[0]

                                if int(a) > int(b):
                                       sensor_data[element].append("mean(grass_temperature PT1H)")
                                       sensor_data[element].append(x["timeResolution"])
                                       sensor_data[element].append(x["sourceId"])


                except:
                        sensor_data[element].append("mean(grass_temperature PT1H)")
                        sensor_data[element].append(x["timeResolution"])
                        sensor_data[element].append(x["sourceId"])
                

        if x["elementId"] == "cloud_area_fraction" and (x["timeResolution"] == "PT1H" or x["timeResolution"] == "PT30M" or x["timeResolution"] == "PT10M" or x["timeResolution"] == "PT6H"):

                try: 
                        if x["validTo"]:
                                a = x["validTo"].split("-")[0]
                                b = data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1].split("-")[0]

                                if int(a) > int(b):
                                       sensor_data[element].append("cloud_area_fraction")
                                       sensor_data[element].append(x["timeResolution"])
                                       sensor_data[element].append(x["sourceId"])


                except:
                        sensor_data[element].append("cloud_area_fraction")
                        sensor_data[element].append(x["timeResolution"])
                        sensor_data[element].append(x["sourceId"])


download_data = {}

dd = []

s = [x for x in sensor_data]
print(len(s))

counter = 0
for element in sensor_data:
        if "sum(precipitation_amount PT1H)" in sensor_data[element] and "air_temperature" in sensor_data[element]: #"air_temperature" in sensor_data[element] #and "sum(precipitation_amount PT1H)" in sensor_data[element]:
                
                dd.append(sensor_data[element])
                #print(sensor_data[element])
                
                download_data[element] = {}


                period = get_period(data.loc[data["vegobjektid"] == element]["Ulykkesdato"].to_string().split(" ")[-1])
                download_data[element]["ulykkesperiode"] = period
                download_data[element]["id"] = element
                #download_data[element]["data"] = {}
                #download_data[element]["data"] = {}
                download_data[element]["sensors"] = {}
                for x in range(0, len(sensor_data[element]), 3):
                        #sensor_data[element][x], sensor_data[element][x+1], sensor_data[element][x+2]
                        try: 
                                download_data[element]["sensors"][sensor_data[element][x+2].split(":")[0]]
                        except:        
                                download_data[element]["sensors"][sensor_data[element][x+2].split(":")[0]] = {}

                        temp_str = "https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements={}&timeresolutions={}".format(sensor_data[element][x+2].split(":")[0], period, sensor_data[element][x], sensor_data[element][x+1])
                        
                        download_data[element]["sensors"][sensor_data[element][x+2].split(":")[0]][sensor_data[element][x]] = temp_str

                        #download_data[element][sensor_data[element][x]].append(temp_str)

                #print(sensor_data[element], element)

                #download_data[element] = 

#for element in download_data:
#        if len(download_data[element]['sensors']) > 1:
#                print(download_data[element]['sensors'])

print(len(dd))

#print(download_data)

output = open('sensor_list.pkl', 'wb')
pickle.dump(download_data, output)
output.close()
