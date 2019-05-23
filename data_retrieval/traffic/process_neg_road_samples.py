
"""
 Read csv containing valid negative road refs. 
 Select first for each accident. 
 Populate road features.
"""

import ast
import asyncio
import time

import aiohttp
import motor.motor_asyncio
import pandas as pd
from pyproj import Proj, transform

import dataset_func as dfun

start_time = time.time()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

client = motor.motor_asyncio.AsyncIOMotorClient(
    'mongodb://***:27017', username='***', password='***', io_loop=loop)
collection = client.veg.lenker_single_validroads  # 2

data = pd.read_csv('dump2_processed.csv', sep=';',
                   encoding='latin-1', dtype='str')
approved_refs = pd.read_csv('test_res.csv', sep=';',
                            encoding='latin-1', dtype='str')

print('original data len:', len(data))


def elapsed():
    return '{:.2f};'.format(time.time() - start_time)


approved_refs_dict = {}
counter = 0
for i, row in approved_refs.iterrows():
    approved = ast.literal_eval(row['Column_with_data'])
    #row_vegobjektid = row['vegobjektid']
    row_vegobjektid = row['vegobjektid']
    if len(approved) == 0:
        i_drop = data[data.vegobjektid == row_vegobjektid].index
        data.drop(i_drop, inplace=True)
        counter += 1
        if counter % 500 == 0:
            print(counter)
        continue

    # Select firts ref
    selected_ref = approved[0]
    approved_refs_dict[row['vegobjektid']] = {'ulykkestidspunkt': row['Ulykkestidspunkt'],
                                              'ulykkesdato': row['Ulykkesdato'],
                                              'vegref_kortform': row['vegref_kortform'],
                                              'vegref': row[f'vegref{selected_ref}'],
                                              'vegref_m': row[f'vegref_m{selected_ref}'],
                                              'geometri': row[f'geometri{selected_ref}']}


def replace_vegref(row):
    return approved_refs_dict[row['vegobjektid']]['vegref']


def replace_vegref_m(row):
    return approved_refs_dict[row['vegobjektid']]['vegref_m']


def replace_geometri_point(row):
    return approved_refs_dict[row['vegobjektid']]['geometri']


neg_road_df = data.copy()
neg_road_df = neg_road_df[neg_road_df.sample_type == 'positive']
"""
 Fields to replace:
  - vegref_kortform
  - vegreferanse 
  - geometri (punkt)
  - geometri (line)
  - kommune
  - veglengde
  - num_cross
  - felt
  - speedlimit
  - road_type
  - road_air_distance
  - curve
"""
neg_road_df.label = '0'
neg_road_df.sample_type = 'road'
neg_road_df['vegref_kortform'] = neg_road_df.apply(replace_vegref, axis=1)
neg_road_df['vegreferanse'] = neg_road_df.apply(replace_vegref_m, axis=1)
neg_road_df['geometri'] = neg_road_df.apply(replace_geometri_point, axis=1)


def extract_road_length(row):
    ref = row['vegref_kortform']
    temp = ref.split()[-1].replace('m', '').split('-')
    fra = temp[0]
    try:
        til = temp[1]
    except:
        til = 0
    return fra, til, int(til) - int(fra)


async def main():
    print(elapsed() + 'fix road length')
    # neg_road_df['veglengde'] = neg_road_df.apply(extract_road_length, axis=1)
    temp = [extract_road_length(e) for _, e in neg_road_df.iterrows()]
    fra = [e[0] for e in temp]
    til = [e[1] for e in temp]
    lengde = [e[2] for e in temp]
    neg_road_df['fra'] = fra
    neg_road_df['til'] = til
    neg_road_df['veglengde'] = lengde

    # do mongo stuff here
    print(elapsed() + 'starting geom bulk')
    geom = await asyncio.gather(
        *[dfun.get_road_line_geometri_and_length(c, collection) for _, c in neg_road_df.iterrows()]
    )

    neg_road_df['veg_geometri'] = [e[0] for e in geom]
    neg_road_df['fylkesnummer'] = [e[1] for e in geom]
    neg_road_df['kommunenummer'] = [e[2] for e in geom]

    # breakpoint()
    async with aiohttp.ClientSession() as session:
        print(elapsed(), 'starting cross')
        crosses = pd.Series(await asyncio.gather(
            *[dfun.calc_num_cross(c, session) for _, c in neg_road_df.iterrows()]
        ))
        neg_road_df['num_cross'] = crosses

        print(elapsed(), 'starting felt')
        felts = pd.Series(await asyncio.gather(
            *[dfun.get_num_felt(c, session) for _, c in neg_road_df.iterrows()]
        ))
        neg_road_df['felt'] = felts

        print(elapsed(), 'starting speedlimit')
        speedlimits = pd.Series(await asyncio.gather(
            *[dfun.get_speedlimit_objects(c, session) for _, c in neg_road_df.iterrows()]
        ))
        neg_road_df['speedlimit'] = speedlimits

        print(elapsed(), 'starting road_type')
        road_types = pd.Series(await asyncio.gather(
            *[dfun.get_road_type(c, session) for _, c in neg_road_df.iterrows()]
        ))
        neg_road_df['road_type'] = road_types

        # fix kommune
        print(elapsed(), 'starting kommune')
        kommune_names = pd.Series(await asyncio.gather(
            *[dfun.get_kommune_name(c, session) for _, c in neg_road_df.iterrows()]
        ))
        neg_road_df['kommune'] = kommune_names

loop.run_until_complete(main())
# breakpoint()
print(elapsed() + 'starting road_air_distance')
neg_road_df['road_air_distance'] = 'temp'
neg_road_df['road_air_distance'] = neg_road_df.apply(
    dfun.calc_road_air_dist, axis=1)

print(elapsed() + 'starting curve')
neg_road_df['curve'] = 'temp'
neg_road_df['curve'] = neg_road_df.apply(dfun.calc_curve, axis=1)

neg_road_df.to_csv('neg_roads.csv', sep=';', encoding='latin-1')

print(len(data))
print(counter)
