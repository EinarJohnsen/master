import aiohttp
import asyncio
import pandas as pd
import transform_epsg_coords
import polygon_circle
import time
import pickle
import logging

logging.basicConfig(filename='download.log',level=logging.INFO)


temp_data = pd.read_csv("path_to_accidents.csv",sep=";", encoding="latin-1")


async def get_frost_data(gemoetry_50_stykker, session):
    
        async with session.get('https://frost.met.no/sources/v0.jsonld?geometry={}'.format(gemoetry_50_stykker)) as resp:
            try: 
                r = await resp.json()
                #print(r["data"][0]["id"], r["totalItemCount"])
                return [r["data"][i]["id"] for i in range(len(r["data"]))]
            except:
                #print(resp)
                return None


async def get_air_temp(sensor, session, time):
        ddd = "https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=air_temperature&timeresolutions=PT1H, PT30M, PT10M".format(sensor, time)
        #print(ddd)
        async with session.get('https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=air_temperature&timeresolutions=PT1H, PT30M, PT10M'.format(sensor, time)) as resp:
            #print(resp)
            try: 
                r = await resp.json()
                #print(r["data"][0]["id"], r["totalItemCount"])
                #print(r)
                return r["data"]
            except:
                #print(resp)
                return None


async def get_nedbor(sensor, session, time):
        #ddd = "https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=air_temperature&timeresolutions=PT1H, PT30M, PT10M".format(sensor, time)
        #print(ddd)
        async with session.get("https://frost.met.no/observations/v0.jsonld?sources={}&referencetime={}&elements=sum(precipitation_amount PT1H), sum(precipitation_amount PT30M), sum(precipitation_amount PT10M)&timeresolutions=PT1H, PT10M, PT30M".format(sensor, time)) as resp:
            #print(resp)
            try: 
                r = await resp.json()
                #print(r["data"][0]["id"], r["totalItemCount"])
                #print(r)
                return r["data"]
            except:
                #print(resp)
                return None
        

async def main_loop():
    data_dict = {}
    auth = aiohttp.BasicAuth(login="FROST AUTH KEY", password="")
    async with aiohttp.ClientSession(auth=auth) as session:
        start_time = time.time()
        counter = 0
        for _tasks in tasks:   
            counter += 1          
            data_dict[_tasks] = {}
            sensors = await asyncio.gather(*[get_frost_data(x, session) for x in tasks[_tasks]["tasks"]])
            #print(sensors)
            air_temp = await asyncio.gather(*[get_air_temp(','.join(x), session, tasks[_tasks]["date"]) for x in sensors if x != None])
            _sensors = [x for x in sensors if x != None]
            _air_temp_data = [_sensors[i] for i in range(len(_sensors)) if air_temp[i] != None]
            perticipation = await asyncio.gather(*[get_nedbor(','.join(x), session, tasks[_tasks]["date"]) for x in _air_temp_data])
            #print(perticipation)

            data_dict[_tasks]["air_temp"] = [x for x in air_temp if x != None]
            data_dict[_tasks]["perticipation"] = [x for x in perticipation if x != None]

            #print(data)
            #data_2 = await asyncio.gather(*[get_frost_data_2(x, session) for x in tasks])

            if counter % 10 == 0:
                output = open('data/sensor_{}.pkl'.format(counter), 'wb')
                pickle.dump(data_dict, output)
                output.close()
                data_dict = {}

                logging.info(str(time.time() - start_time) + "  ---  " + str(counter))

tasks = {}

for c in temp_data.values:
    mix_data = [c[1], c[2], c[3]]
    data = c[-50:]
    tasks[mix_data[0]] = {}
    polygons = []
    for element in data:
        if element != "ukjent":
            try:
                point1,point2 = element.split(" ")[2][1:],element.split(" ")[3]
            except:
                point1, point2 = element.split(" ")[1][1:], element.split(" ")[2][:-1]

            point1, point2 = transform_epsg_coords.transform_coords(point1, point2)
            polygon = polygon_circle.get_frost_polygon(point1, point2)

            polygons.append(polygon)
   
    tasks[mix_data[0]]["tasks"] = polygons 
    dd = ''.join(str(mix_data[2])+"T"+str(mix_data[1])+":00"+"/"+str(mix_data[2])+"T"+str(int(mix_data[1].split(":")[0])+1)+":00:00")
    tasks[mix_data[0]]["date"] = dd

loop = asyncio.get_event_loop()
loop.run_until_complete(main_loop())

