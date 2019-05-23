
import asyncio
import random
import re
import time

import aiohttp
import motor.motor_asyncio
import pandas as pd

import util

# Make sure only one loop is active
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# url and header
veg_url = 'https://www.vegvesen.no/nvdb/api/v2/veg?vegreferanse={}'
headers = {'X-Client': 'Master student - UiA', 'X-Kontaktperson': '***'}

# Mongo
client = motor.motor_asyncio.AsyncIOMotorClient(
    'mongodb://***:27017', username='***', password='***', io_loop=loop)
collection = client.veg.lenker_single_validroads

data = pd.read_csv('dump2.csv', sep=';', encoding='latin-1').dropna()
data = data[['vegobjektid', 'Ulykkestidspunkt',
             'Ulykkesdato', 'vegref_kortform']]
data = data[:10]

print('starting timer...')
start_time = time.time()


def elapsed():
    return '{:.2f};'.format(time.time() - start_time)


async def do_samples(ref):
    samples = []
    cursor = collection.aggregate(util.sample_query)
    for d in await cursor.to_list(length=5):
        samples.append(d)
    return samples


async def get_road_point_geometry(vegreferanse):
    async with aiohttp.ClientSession() as session:
        async with session.get(veg_url.format(vegreferanse), headers=headers) as resp:
            r = await resp.json()
            return r['geometri']['wkt']


async def main():
    print('Go loop!')
    samples = await asyncio.gather(
        *[do_samples(ref) for ref in data['vegref_kortform']]
    )
    print(elapsed() + 'done sampling!')
    vegref = {i: list() for i in range(util.num_samples)}
    vegref_m = {i: list() for i in range(util.num_samples)}
    geometry = {i: list() for i in range(util.num_samples)}

    for obj in samples:
        for i, s in enumerate(obj):
            if data.iloc[i]['vegref_kortform'] == s['vegreferanse']['kortform']:
                vegref[i].append('invalid')
                vegref_m[i].append('invalid')
                geometry[i].append('invalid')
            rand_meter = random.randint(
                s['vegreferanse']['fra_meter'], s['vegreferanse']['til_meter'])
            road_point = re.sub(r'(\d*-\d*)', str(rand_meter),
                                s['vegreferanse']['kortform'])
            vegref[i].append(s['vegreferanse']['kortform'])
            vegref_m[i].append(road_point)
    for k, v in vegref.items():
        data[f'vegref{k}'] = v
    for k, v in vegref_m.items():
        data[f'vegref_m{k}'] = v
    print(elapsed() + 'done pandas building!')
    for i in range(util.num_samples):
        point_geoms = pd.Series(await asyncio.gather(
            *[get_road_point_geometry(r) for r in data[f'vegref_m{i}']]
        ))
        data[f'geometri{i}'] = point_geoms
    print(elapsed() + 'done geometry querying!')


# loop.set_debug(True)
loop.run_until_complete(main())
