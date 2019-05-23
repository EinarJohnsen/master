import pickle
import pandas as pd 
import polygon_circle
import transform_epsg_coords
import shapely.geometry
from shapely.geometry import Point, Polygon
import datetime

temp_data = pd.read_csv("path_to_negative_roads.csv",sep=";", encoding="latin-1")
#for x in temp_data:
#    print(x)
temp_data2 = temp_data[['vegobjektid', 'geometri', 'Ulykkesdato', 'Ulykkestidspunkt']]

temp_data3 = temp_data[['Ulykkesdato']]
ddd = [x[0].split("-")[0] for x in temp_data3.values]
print(sorted(set(ddd)))


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

pkl_file = open('d_all.pkl', 'rb')                                                                                                                                                                                           
rows_with_data = pickle.load(pkl_file)                                                                                                                                                                                                          
pkl_file.close()

pkl_file = open('sensor_coordinates.pkl', 'rb')                                                                                                                                                                                           
sensor_coords = pickle.load(pkl_file)                                                                                                                                                                                                          
pkl_file.close()

data_dict = {}
big_list_1 = []
big_list_2 = []
counter = 0 

temp_data = temp_data.reset_index(drop=True)


print(len(temp_data2))

new_weather_dict = {}

for ele in temp_data2.values:
    if counter == 10:
        break 
    counter += 1 

    try:
        point1,point2 = ele[1].split(" ")[2][1:],ele[1].split(" ")[3]
    except:
        point1, point2 = ele[1].split(" ")[1][1:], ele[1].split(" ")[2][:-1]

    point1, point2 = transform_epsg_coords.transform_coords(point1, point2)
    #p = shapely.geometry.Point([point1, point2])
    polygon = polygon_circle.get_frost_polygon_2(point1, point2)

    try:
        data = rows_with_data[ele[0]]
    except:
        pass 
    """
    for x in data["air_temp"]:
        print(x)
        print(" ")
    print("**************************************")
    for y in data["perticipation"]:
        print(y)
        print(" ")
    """
    #print("**************************************")
    
    #print(data)

    air_temp_sensors = [x[0]["sourceId"] for x in data["air_temp"]]
    perc_sensors = [x[0]["sourceId"] for x in data["perticipation"]]

    air_temp_sensors = list(set(air_temp_sensors))
    perc_sensors = list(set(perc_sensors))
    #print(polygon)

    inside_air = []
    for x in air_temp_sensors:
        points = sensor_coords[x.split(":")[0]]
        p = shapely.geometry.Point([points[0], points[1]])
        if p.within(polygon):
            inside_air.append(x)
        

    inside_pec = []
    for y in perc_sensors:
        points = sensor_coords[y.split(":")[0]]
        p = shapely.geometry.Point([points[0], points[1]])
        if p.within(polygon):
            inside_pec.append(y)

    #print(inside_air, " air")
    #print(inside_pec, " pec")
    #print(ele)

    new_weather_dict[ele[0]] = {}
    new_weather_dict[ele[0]]["at"] = inside_air
    new_weather_dict[ele[0]]["p"] = inside_pec
    new_weather_dict[ele[0]]["tid"] = ele[3]
    new_weather_dict[ele[0]]["dato"] = ele[2]
    new_weather_dict[ele[0]]["gmt"] = date_checker(ele[2])

    if (int(ele[3].split(":")[0])+date_checker(ele[2]))%24 == 0: 

        time = (int(ele[3].split(":")[0])+date_checker(ele[2]))%24
        
        try:
            temp = ele[2].split("-")
            next_date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2])+1, time)
            next_date += datetime.timedelta(hours=1)
        except:
            next_date = datetime.datetime(int(temp[0]), int(temp[1])+1, 1, time)
            next_date += datetime.timedelta(hours=1)
            time = "00"
            ele[2] = datetime.datetime(int(temp[0]), int(temp[1])+1, 1)
    
    else:
        time = int(ele[3].split(":")[0])+date_checker(ele[2])
        try:
            temp = ele[2].split("-")
            next_date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2]), time)
            next_date += datetime.timedelta(hours=1)
        except:
            print("Not a date")

    inside_air = list(set(inside_air))
    inside_pec = list(set(inside_pec))
    
    ref_time = str(ele[2]) + "T" + str(time) +":00:00/" + str(next_date).split(" ")[0] + "T" + str(next_date).split(" ")[1]

    defg = ','.join(inside_air + inside_pec)
    d_str = "https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=air_temperature,sum(precipitation_amount PT1H)".format(defg, ref_time)

    new_weather_dict[ele[0]]["ref_time"] = ref_time
    new_weather_dict[ele[0]]["link"] = d_str


print(new_weather_dict)



