
import asyncio
import calendar
import datetime
import random
import re

import aiohttp
import backoff
import requests
from geopy.distance import great_circle
from pyproj import Proj, transform

headers = {'X-Client': 'Master student - UiA', 'X-Kontaktperson': '***'}
cross_url = 'https://www.vegvesen.no/nvdb/api/v2/vegobjekter/37?vegreferanse={}'
felt_url = 'https://www.vegvesen.no/nvdb/api/v2/vegnett/lenker?vegreferanse={}'
speedlimit_url = 'https://www.vegvesen.no/nvdb/api/v2/vegobjekter/105?vegreferanse={}'
veg_url = 'https://www.vegvesen.no/nvdb/api/v2/veg?vegreferanse={}'
kommune_url = 'https://www.vegvesen.no/nvdb/api/v2/vegobjekter/536?vegreferanse={}'


def replace_hour(h):
    while True:
        rand_int = str(random.randint(0, 23))
        if len(rand_int) == 1:
            rand_int = '0' + rand_int
        rand_hour = rand_int + ':00'

        if rand_hour != h:
            return rand_hour


def calc_road_air_dist(row):
    inProj = Proj(init='epsg:32633')
    outProj = Proj(init='epsg:4326')

    points = re.search(
        r'(?:LINESTRING.*\()([\d,\s.-]*)', row['veg_geometri'])[1].split(',')
    start = points[0].split()
    end = points[-1].split()
    lon_start, lat_start = transform(inProj, outProj, start[0], start[1])
    lon_end, lat_end = transform(inProj, outProj, end[0], end[1])
    return great_circle((lat_start, lon_start), (lat_end, lon_end)).meters


def calc_curve(row):
    try:
        return float(row['veglengde']) / float(row['road_air_distance'])
    except ZeroDivisionError:
        return 0


async def get_road_point_geometry(row):
    async with aiohttp.ClientSession() as session:
        async with session.get(veg_url.format(row['vegreferanse']), headers=headers) as resp:
            r = await resp.json()
            return r['geometri']['wkt']


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def calc_num_cross(row, session):
    async with session.get(cross_url.format(row['vegref_kortform']), headers=headers) as resp:
        r = await resp.json()
        return r['metadata']['returnert']


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def get_num_felt(row, session):
    async with session.get(felt_url.format(row['vegreferanse']), headers=headers) as resp:
        r = await resp.json()
        for obj in r['objekter']:
            if obj['geometri']['wkt'] == row['veg_geometri']:
                try:
                    return obj['felt']
                except KeyError:
                    return 'ukjent'
        return 'ukjent'


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def get_road_type(row, session):
    async with session.get(felt_url.format(row['vegreferanse']), headers=headers) as resp:
        r = await resp.json()
        for obj in r['objekter']:
            if obj['geometri']['wkt'] == row['veg_geometri']:
                return obj['typeVeg']
        return 'ukjent'


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def get_speedlimit_objects(row, session):
    async with session.get(speedlimit_url.format(row['vegreferanse']), headers=headers) as resp:
        r = await resp.json()
        slimits = await asyncio.gather(
            *[get_speedlimit(obj['href'], session) for obj in r['objekter']]
        )
        if len(set(slimits)) == 1:
            return slimits[0]
        else:
            return 'ukjent'


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def get_speedlimit(url, session):
    async with session.get(url, headers=headers) as resp:
        r = await resp.json()
        for e in r['egenskaper']:
            if e['navn'] == 'Fartsgrense':
                return e['verdi']
        return 'ukjent'


def insert_day(row):
    day = datetime.datetime.strptime(row['Ulykkesdato'], r'%Y-%m-%d')
    return int(day.timetuple().tm_yday)


def generate_day(d):
    day = datetime.datetime.strptime(d, r'%Y-%m-%d')
    month = random.randint(1, 12)
    max_day = max(calendar.monthrange(day.year, month))
    rand_day = day.replace(month=month, day=random.randint(1, max_day))
    return str(rand_day.date())


def generate_month(row):
    date = datetime.datetime.strptime(row['Ulykkesdato'], r'%Y-%m-%d')
    return date.month


def date_to_weekday(d):
    days = {0: 'Mandag', 1: 'Tirsdag', 2: 'Onsdag',
            3: 'Torsdag', 4: 'Fredag', 5: 'Lørdag', 6: 'Søndag'}
    d = datetime.datetime.strptime(d, r'%Y-%m-%d')
    return days[d.timetuple().tm_wday]


def get_road_geometry(row, coll):
    ref = re.search(r'(\d*\s{1}\w*\d*\s{1}hp\d*)', row.vegreferanse)[0]
    doc = coll.find_one({
        '$and': [
            {
                'vegreferanse.kortform': re.compile(r'{}'.format(ref))
            }, {
                'vegreferanse.fra_meter': {
                    '$lte': int(row['fra meter'])
                }
            }, {
                'vegreferanse.til_meter': {
                    '$gte': int(row['fra meter'])
                }
            }
        ]
    })
    try:
        fra = doc['vegreferanse']['fra_meter']
        til = doc['vegreferanse']['til_meter']
    except TypeError:
        return 'ukjent', 'ukjent', 'ukjent', 'ukjent', 'ukjent'
    return doc['geometri']['wkt'], doc['vegreferanse']['kortform'], fra, til, int(til) - int(fra)


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60)
async def get_kommune_name(row, session):
    knr = re.match(r'\d{4}', row['vegref_kortform'])[0]
    async with session.get(kommune_url.format(knr), headers=headers) as resp:
        r = await resp.json()
        async with session.get(r['objekter'][0]['href'], headers=headers) as resp:
            r = await resp.json()
            for e in r['egenskaper']:
                if e['navn'] == 'Kommunenavn':
                    return e['verdi']
            return 'ukjent'


async def get_road_line_geometri_and_length(row, coll):
    ref = re.search(r'(\d*\s{1}\w*\d*\s{1}hp\d*)', row.vegreferanse)[0]
    doc = await coll.find_one({
        '$and': [
            {
                'vegreferanse.kortform': re.compile(r'{}'.format(ref))
            }, {
                'vegreferanse.fra_meter': {
                    '$lte': int(row['fra'])
                }
            }, {
                'vegreferanse.til_meter': {
                    '$gte': int(row['fra'])
                }
            }
        ]
    })

    fylkesnummer = doc['vegreferanse']['fylke']
    kommunenummer = doc['vegreferanse']['kommune']
    return doc['geometri']['wkt'], fylkesnummer, kommunenummer
