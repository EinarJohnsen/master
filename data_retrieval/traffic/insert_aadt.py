
# Insert AADT based on 'vegref_kortform'

import asyncio
import time
from collections import namedtuple

import aiohttp
import backoff
import pandas as pd

headers = {'X-Client': 'Master student - UiA', 'X-Kontaktperson': '***'}
aadt_url = 'https://www.vegvesen.no/nvdb/api/v2/vegobjekter/540?vegreferanse={}'

data = pd.read_csv('dump2_processed.csv', sep=';', encoding='latin-1')
data = data[:1000]

# Start timer
start_time = time.time()
print('Starting timer...')


async def get_aadt(url, vref, session):
    Result = namedtuple(
        'Result', ['aadt', 'year_valid', 'percentage_long_vehicles'])
    async with session.get(url, headers=headers) as resp:
        r = await resp.json()
        year_valid = 'ukjent'
        aadt = 'ukjent'
        percentage_long_vehicles = 'ukjent'
        for e in r['egenskaper']:
            if e['id'] == 4621:
                year_valid = e['verdi']
            if e['id'] == 4623:
                aadt = e['verdi']
            if e['id'] == 4624:
                percentage_long_vehicles = e['verdi']
        return Result(aadt=aadt, year_valid=year_valid, percentage_long_vehicles=percentage_long_vehicles)


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def get_aadt_objects(row, session):
    async with session.get(aadt_url.format(row['vegref_kortform']), headers=headers) as resp:
        r = await resp.json()
        if r['metadata']['returnert'] == 0:
            return 'ukjent'
        aadts = await asyncio.gather(
            *[get_aadt(o['href'], row['vegref_kortform'], session) for o in r['objekter']]
        )
        return aadts


async def main():
    async with aiohttp.ClientSession() as session:
        aadt = await asyncio.gather(
            *[get_aadt_objects(r, session) for _, r in data.iterrows()]
        )
    data['aadt'] = aadt

asyncio.run(main())
print('Elapsed', time.time() - start_time)

# Extract aadt and large vehicle feature, and calculate average if needed
aadt, percentage_long_vehicles = [], []
for _, r in data.iterrows():
    if r.aadt == 'ukjent':
        aadt.append('ukjent')
        percentage_long_vehicles.append('ukjent')
        continue
    if len(r.aadt) > 1:
        print(r.aadt)
        newest_year = max(r.aadt, key=lambda r: r.year_valid).year_valid
        newest_aadt = [a for a in r.aadt if a.year_valid == newest_year]
        print(newest_year)
        print(newest_aadt)
