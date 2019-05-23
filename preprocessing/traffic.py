
""" Provides <load_traffic> func for loading dataset """

import calendar
import re
from collections import OrderedDict, namedtuple
from itertools import chain

import munch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)


def load_traffic(pipeline='basic', filepath='train_raw.csv', scaler=MinMaxScaler, scale=True, spectral_path=None):
    """ Load traffic dataset. 
        Basic pipeline only converts <lanes>.
        Full pipeline converts all features and scales them (for dnn) """

    if pipeline not in ['basic', 'full', 'numeric']:
        raise ValueError('pipeline arg must be basic or full')

    # Including target
    org_features = ('day', 'month', 'Ukedag', 'Ulykkestidspunkt', 'veglengde', 'num_cross', 'road_type', 'felt',
                    'speedlimit', 'curve', 'air_temp', 'precipitation', 'aadt', 'percentage_long_vehicles', 'label')

    feature_name_mappings = {
        'Ukedag': 'weekday',
        'Ulykkestidspunkt': 'hour',
        'veglengde': 'road_length',
        'felt': 'lanes',
    }

    weekday_mappings = {
        'Mandag': 'monday',
        'Tirsdag': 'tuesday',
        'Onsdag': 'wednesday',
        'Torsdag': 'thursday',
        'Fredag': 'friday',
        'Lørdag': 'saturday',
        'Søndag': 'sunday',
    }

    road_type_mappings = {
        'enkelBilveg': 'ordenary_road',
        'kanalisertVeg': 'channeled_road',
        'rundkjøring': 'roundabout',
    }

    numeric_features = ['road_length', 'num_cross', 'curve',
                        'air_temp', 'precipitation', 'aadt', 'percentage_long_vehicles']
    categorical_features = ['road_type']
    cyclic_features = ['day', 'month', 'weekday', 'hour']
    ordinal_features = ['speedlimit']

    # Read raw dataset
    data_raw = pd.read_csv(filepath, sep=';', encoding='latin-1')

    data = data_raw.copy()[[*org_features, 'sample_type']]

    # spectral features from txt
    if spectral_path is not None:
        spectral_features = pd.read_csv(
            spectral_path, sep=' ', dtype=float, header=None)
        spectral_features.columns = [
            f'spec{n}' for n in range(len(spectral_features.columns))]
        # add spectral features to data
        data = data.join(spectral_features)

    # Clean up column names
    data.rename(index=str, columns=feature_name_mappings, inplace=True)

    # Turn norwegian categorical values into english
    data.replace(to_replace=weekday_mappings, inplace=True)
    data.replace(to_replace=road_type_mappings, inplace=True)

    # Remove negative precipitation, except -1 (will be set to 0). Ref: mail from yr.no
    data.replace({'precipitation: {-1: 0}'}, inplace=True)
    data = data[data.precipitation >= 0]
    data.reset_index(inplace=True, drop=True)

    # Column names
    columns = list(data.columns)

    # Divide data. Each will have their own pipeline
    data_num = data[numeric_features]
    data_cat = data[categorical_features]
    data_lane = data[['lanes']]
    data_cyclic = data[cyclic_features]
    data_ordinal = data[ordinal_features]

    if spectral_path is not None:
        data_spectral = data[spectral_features.columns]

    # get sample type and pass outside data
    sample_type = data['sample_type']

    # numeric is intended for visualization
    if pipeline == 'basic' or pipeline == 'numeric':
        data_lane_pipeline = make_pipeline(
            _LaneEncoder(),
        )
        data_lane_transformed = data_lane_pipeline.fit_transform(data_lane)
        columns.remove('lanes')
        columns.remove('label')
        data_targets = data.label
        data.drop(columns=['lanes', 'label'], inplace=True)
        columns.extend(['num_ord_lanes',
                        'num_bus_lanes',
                        'num_turn_lanes',
                        'num_bike_lanes',
                        'num_pitch_lanes',
                        'num_rev_lanes']),
        columns.append('target')
        if pipeline == 'basic':
            return munch.Munch(data=np.hstack((np.array(data), data_lane_transformed)),
                               feature_names=columns,
                               target=data_targets.to_numpy())
        else:
            data.replace({'weekday': {v: k for k, v in enumerate(
                [d.lower() for d in calendar.day_name])}}, inplace=True)
            data.replace({'hour': {v: k for k, v in enumerate(
                sorted(data.hour.unique()))}}, inplace=True)
            data.day = data.day - 1
            data.month = data.month - 1
            data.replace({'road_type': {v: k for k, v in enumerate(
                sorted(data.road_type.unique()))}}, inplace=True)
            return munch.Munch(data=np.hstack((np.array(data), data_lane_transformed)),
                               feature_names=columns,
                               target=data_targets.to_numpy())

    elif pipeline == 'full':
        # TODO: Should we use StandardScaler or MinMaxScaler?
        data_cyclic_pipeline = make_pipeline(
            _CyclicEncoder(),
            scaler() if scale == True else None
        )
        data_lane_pipeline = make_pipeline(
            _LaneEncoder(),
            scaler() if scale == True else None
        )
        data_num_pipeline = make_pipeline(
            scaler() if scale == True else None
        )
        data_cat_pipeline = make_pipeline(
            OneHotEncoder()
        )
        data_ord_pipeline = make_pipeline(
            OrdinalEncoder(),
            scaler() if scale == True else None
        )
        if spectral_path is not None:
            data_spectral_pipeline = make_pipeline(
                scaler() if scale == True else None
            )
            data_spectral_transformed = data_spectral_pipeline.fit_transform(
                data_spectral)
        data_cyclic_transformed = data_cyclic_pipeline.fit_transform(
            data_cyclic)
        data_lane_transformed = data_lane_pipeline.fit_transform(data_lane)
        data_num_transformed = data_num_pipeline.fit_transform(data_num)
        data_cat_transformed = data_cat_pipeline.fit_transform(data_cat)
        data_ord_transformed = data_ord_pipeline.fit_transform(data_ordinal)
        feature_names = []
        feature_names.extend(chain.from_iterable(
            ('{}_x'.format(f), '{}_y'.format(f)) for f in cyclic_features))
        feature_names.extend(numeric_features)
        feature_names.extend(data_cat_pipeline.steps[0][1].get_feature_names())
        feature_names.extend(['num_ord_lanes',
                              'num_bus_lanes',
                              'num_turn_lanes',
                              'num_bike_lanes',
                              'num_pitch_lanes',
                              'num_rev_lanes'])
        feature_names.extend(ordinal_features)
        if spectral_path is not None:
            feature_names.extend(spectral_features.columns)
            return munch.Munch(data=np.hstack((data_cyclic_transformed, data_num_transformed, data_cat_transformed.todense(), data_lane_transformed, data_ord_transformed, data_spectral_transformed)),
                               target=data.label.values, org=data, feature_names=feature_names, sample_type=sample_type)

        return munch.Munch(data=np.hstack((data_cyclic_transformed, data_num_transformed, data_cat_transformed.todense(), data_lane_transformed, data_ord_transformed)),
                           target=data.label.values, org=data, feature_names=feature_names, sample_type=sample_type)


class _LaneEncoder(BaseEstimator, TransformerMixin):
    """ 
        Transform lane info from NPRA into six features:
        0. num_ord_lanes (ordinary lanes)
        1. num_bus_lanes (bus lanes)
        2. num_turn_lanes (right and left turn lanes)
        3. num_bike_lanes (bike lanes)
        4. num_pitch_lanes ("oppstillingsplass")
        5. num_rev_lanes (reverable lanes)
    """
    @staticmethod
    def _extract_lanes(lane_string):
        lanes = [0 for n in range(6)]
        lane_elements = lane_string.lower().split('#')
        for e in lane_elements:
            if 'k' in e:
                lanes[1] += 1
            elif 'v' in e or 'h' in e:
                lanes[2] += 1
            elif 's' in e:
                lanes[3] += 1
            elif 'o' in e:
                lanes[4] += 1
            elif 'r' in 'e':
                lanes[5] += 1
            else:
                lanes[0] += 1
        return lanes

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([self._extract_lanes(ls) for i in range(len(X.columns))
                         for ls in X[X.columns[i]]])


class _CyclicEncoder(BaseEstimator, TransformerMixin):
    """ Encodes cyclic features as x, y coords in a circle """

    def fit(self, X):
        weekdays = [d.lower() for d in calendar.day_name]
        Feature = namedtuple('Feature', 'num_unique mapping')
        self.features = OrderedDict((X.columns[i], Feature(len(X[X.columns[i]].unique()),
                                                           {v: k for k, v in enumerate(sorted(set(X[X.columns[i]])))}))
                                    for i in range(len(X.columns)))
        # Special handling for weekday
        if 'weekday' in self.features:
            del self.features['weekday']
            self.features['weekday'] = Feature(
                7, {v: k for k, v in enumerate(weekdays)})
        return self

    def _extract_xy(self, cyclic_string, feature):
        x = np.sin(
            2 * np.pi * self.features[feature].mapping[cyclic_string] / self.features[feature].num_unique)
        y = np.cos(
            2 * np.pi * self.features[feature].mapping[cyclic_string] / self.features[feature].num_unique)
        return x, y

    def transform(self, X):
        _, n_features = X.shape
        if len(self.features) != n_features:
            raise ValueError(
                'encoder was fitted for a different set of columns')
        arrays = []
        for feature in self.features:
            a = np.array([self._extract_xy(cs, feature) for cs in X[feature]])
            arrays.append(a)
        return np.hstack(arrays)
