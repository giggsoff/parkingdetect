from pandas import read_csv
from pandas import Series
from pandas import DataFrame
from pandas import datetime
from pandas import concat
from matplotlib import pyplot
import numpy


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


name = 'SKOLEBAKKEN'
dataset = read_csv('data/' + name + '.csv.gz', header=None, index_col=0, parse_dates=[0], date_parser=parser)
print(dataset.head())
upsample = dataset.resample('10min').mean()
print(upsample.head())
interpolated = upsample.interpolate(method='linear', axis=0)
print(interpolated.head())
io = interpolated.round(0).astype(int)
io['Total'] = io.sum(axis=1)
io.to_csv('prepared/'+name+'.csv.gz', header=False, compression='gzip')
