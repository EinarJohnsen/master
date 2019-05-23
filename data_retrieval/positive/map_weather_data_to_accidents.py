import os 
import pickle
import pandas as pd 
import time
import datetime

dates = {}
dates["1987"] = ["29", "25"]
dates["1988"] = ["27", "30"]
dates["1989"] = ["26", "29"]
dates["1990"] = ["25", "28"]
dates["1991"] = ["31", "27"]
dates["1992"] = ["29", "25"]
dates["1993"] = ["28", "31"]
dates["1994"] = ["27", "30"]
dates["1995"] = ["26", "29"]
dates["1996"] = ["31", "27"]
dates["1997"] = ["30", "26"]
dates["1998"] = ["29", "25"]
dates["1999"] = ["28", "31"]
dates["2000"] = ["26", "29"]
dates["2001"] = ["25", "28"]
dates["2002"] = ["31", "27"]
dates["2003"] = ["30", "26"]
dates["2004"] = ["28", "31"]
dates["2005"] = ["27", "30"]
dates["2006"] = ["26", "29"]
dates["2007"] = ["25", "28"]
dates["2008"] = ["30", "26"]
dates["2009"] = ["29", "25"]
dates["2010"] = ["28", "31"]
dates["2011"] = ["27", "30"]
dates["2012"] = ["25", "28"]
dates["2013"] = ["31", "27"]
dates["2014"] = ["30", "26"]
dates["2015"] = ["29", "25"]
dates["2016"] = ["27", "30"]
dates["2017"] = ["26", "29"]
dates["2018"] = ["25", "28"]
dates["2019"] = ["31", "27"]

def date_checker(date):
    date = date.split("-")
    year, month, day = date[0], date[1], date[2]
    #print(year, month, day, dates[year])

    external_date = dates[year]

    gmt = -1
    if int(month) > 3 and int(month) < 10:
        gmt = 2
    elif int(month) == 3 and int(day) < int(external_date[0]):
        gmt = 1
    elif int(month) == 3 and int(day) >= int(external_date[0]):
        gmt = 2 
    elif int(month) == 10 and int(day) < int(external_date[1]):
        gmt = 2 
    elif int(month) == 10 and int(day) >= int(external_date[1]):
        gmt = 1 
    elif int(month) > 10 or int(month) < 3:
        gmt = 1 
    
    return gmt 


def get_meta_info(data, metadata):
    print("-"*50)
    for ele in data:
        for el in data[ele]:  
            for x in data[ele][el]:
                print(x, el)
    
    for ele in data:
        print(ele, "IS ELE")
        for el in data[ele]:        
            #print("*"*50)
            try:
                air_temp = data[ele][el]["air_temperature"]["data"]
            except:
                air_temp = None
                print("none air temp")

            if air_temp != None:
                for x in air_temp:
                    if x['referenceTime'].split("T")[0] == metadata[1]:
                        #print(x)
                        if x['referenceTime'].split("T")[1].split(".")[0].split(":")[0] == metadata[2].split(":")[0]:
                            #air_temp_values.append(x['observations'][0]['value'])
                            print("was inside at", x)
            
            try:
                perticipation = data[ele][el]["sum(precipitation_amount PT1H)"]["data"]
            except:
                perticipation = None

            if perticipation != None:
                for x in perticipation:
                    if x['referenceTime'].split("T")[0] == metadata[1]:
                        if x['referenceTime'].split("T")[1].split(".")[0].split(":")[0] == metadata[2].split(":")[0]:
                            #perticipation_values.append(x['observations'][0]['value'])
                            print("inside", x)
                            
            else: 
                print("none perticipation")
    print("-"*50)
files = os.listdir("path_to_data/")

#elements_in_folder = []

temp_data = pd.read_csv("accidents_to_get_weather.csv",sep=";", encoding="latin-1")
temp_data_filtered = temp_data[['vegobjektid', 'Ulykkesdato', 'Ulykkestidspunkt', 'sample_type', 'geometri']]

#print(temp_data_filtered)


metadata_ele = []
counter =0 
for element in temp_data_filtered.values:
    
    
    time = int(element[2].split(":")[0])+date_checker(element[1])
    
    try:
        temp = element[1].split("-")
        next_date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2]), time)
        #next_date += datetime.timedelta(hours=1)
    except:
        
        if time == 24:
            try:
                temp = element[1].split("-")
                next_date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2])+1, 00)

            except:
                try:
                    next_date = datetime.datetime(int(temp[0]), int(temp[1])+1, 1, 00)

                except:
                    next_date = datetime.datetime(int(temp[0])+1, 1, 1, 00)
                    

        if time == 25:
            try:
                temp = element[1].split("-")
                next_date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2])+1, 1)

            except:
                try:
                    next_date = datetime.datetime(int(temp[0]), int(temp[1])+1, 1, 1)
                   
                except:
                    next_date = datetime.datetime(int(temp[0])+1, 1, 1, 1)

    element[1] = str(next_date).split(" ")[0]
    element[2] = str(next_date).split(" ")[1][0:5]
    metadata_ele.append(element)

    #print(next_date)

    #counter += 1 
    #if counter == 200:
        #metadata = element
    #    break
    #counter += 1 

weather_dict = {}

#print(sorted(set(dddddd)))

def get_data(counter_x):
    for metadata in metadata_ele:
        counter_x += 1 

        if counter_x % 100 == 0:
            print(counter_x)

        try:
            pkl_file = open('path_to_sensor_data/sensor_{}.pkl'.format(metadata[0]), 'rb')
            data = pickle.load(pkl_file)
            pkl_file.close()
        except:
            continue
        
        for ele in data:
            weather_dict[str(ele)+"_"+str(metadata[3])] = {}
            air_temp_values = []
            perticipation_values = []
            for el in data[ele]:        
                try:
                    air_temp = data[ele][el]["air_temperature"]["data"]
                except:
                    air_temp = None

                if air_temp != None:
                    for x in air_temp:
                        if x['referenceTime'].split("T")[0] == metadata[1]:
                            if x['referenceTime'].split("T")[1].split(".")[0].split(":")[0] == metadata[2].split(":")[0]:
                                air_temp_values.append(x['observations'][0]['value'])
                
                try:
                    perticipation = data[ele][el]["sum(precipitation_amount PT1H)"]["data"]
                except:
                    perticipation = None

                if perticipation != None:
                    for x in perticipation:
                        if x['referenceTime'].split("T")[0] == metadata[1]:
                            if x['referenceTime'].split("T")[1].split(".")[0].split(":")[0] == metadata[2].split(":")[0]:
                                perticipation_values.append(x['observations'][0]['value'])
                    
            weather_dict[str(ele)+"_"+str(metadata[3])]["air_temperatures"] = air_temp_values
            weather_dict[str(ele)+"_"+str(metadata[3])]["perticipation"] = perticipation_values

        #get_meta_info(data, metadata)
counter_x = 0 
get_data(counter_x)
#print(weather_dict)
print("Done")
counter2 = 0 
counter3 = 0 
for element in weather_dict: 
    if weather_dict[element]["perticipation"] != []:
        counter2 += 1 
    if weather_dict[element]["air_temperatures"] != []:
        counter3 += 1

print(counter2, counter3, len(weather_dict))

#print(weather_dict)

output = open('nedbor.pkl', 'wb')
pickle.dump(weather_dict, output)
output.close()    


