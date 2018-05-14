from pathlib import Path
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
import matplotlib.dates as mdates
from pandas import datetime
import datetime as dt
import matplotlib.ticker as plticker


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


#name = 'BRUUNS'
#name = 'BUSGADEHUSET'
#name = 'KALKVAERKSVEJ'
#name = 'MAGASIN'
#name = 'NORREPORT'
#name = 'SALLING'
name = 'SCANDCENTER'
#name = 'SKOLEBAKKEN'
# load dataset
dataset = read_csv('prepared/' + name + '.csv.gz', header=None, index_col=0)#, parse_dates=[0], date_parser=parser)
# dataset['Total'] = dataset.sum(axis=1)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, -1] = encoder.fit_transform(values[:, -1])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 5
n_features = dataset.shape[1]
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = int(dataset.shape[0] * 2 / 7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

my_dpi = 96
my_file = Path('results/' + name + '.h5')
# load json and create model
json_file = open('results/' + name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('results/' + name + ".h5")
print("Loaded model from disk")

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, (-n_features + 1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, (-n_features + 1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')
pyplot.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
x = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S') for d in dataset.index.values[:n_train_hours+5]]
#loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
#pyplot.gca().xaxis.set_major_locator(loc)
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
pyplot.ylabel('Occupying %', fontsize=16)
pyplot.xlabel('Date', fontsize=16)
#pyplot.gca().xaxis.set_major_locator(mdates.MinuteLocator())
pyplot.plot(x[:300],test_X[:, -1][:300], label="real")
pyplot.plot(x[:300],yhat[:300], label="predict")
pyplot.axvline(dt.datetime.strptime('2014-05-23 06:30', '%Y-%m-%d %H:%M'), color='r', linestyle='--', lw=2)
pyplot.axhline(y=0.5, color='g', linestyle='--', lw=2)
pyplot.gcf().autofmt_xdate()
#pyplot.plot(dataset.index.to_pydatetime(), dataset.values)
#pyplot.plot(yhat, label="predict")
pyplot.legend()
pyplot.savefig('predict/' + name + "_predict.png", dpi=my_dpi * 5)
pyplot.show()
