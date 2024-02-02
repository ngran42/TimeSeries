import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

import warnings
import logging

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
logging.getLogger('fbprophet').setLevel(logging.ERROR)
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# COMMAND ----------

df = spark.read.format ("csv").load("dbfs:/mnt/customer-analytics/NGrannum/countsv2.csv", header=True)
df.show()

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType


df = df.withColumn("y", df["y"].cast(IntegerType()))
df = df.withColumn("y1", df["y1"].cast(IntegerType()))
df = df.withColumn("y2", df["y2"].cast(IntegerType()))
df = df.withColumn("y3", df["y3"].cast(IntegerType()))
df = df.withColumn("ds", df["ds"].cast(TimestampType()))

# COMMAND ----------

df = spark.read.format ("csv").load("dbfs:/mnt/customer-analytics/NGrannum/counts_p2.csv", header=True)
df.show()

# COMMAND ----------

Pfrom pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType


df = df.withColumn("y", df["y"].cast(IntegerType()))
df = df.withColumn("y1", df["y1"].cast(IntegerType()))
df = df.withColumn("y2", df["y2"].cast(IntegerType()))
df = df.withColumn("y3", df["y3"].cast(IntegerType()))
df = df.withColumn("y4", df["y4"].cast(IntegerType()))
df = df.withColumn("y5", df["y5"].cast(IntegerType()))
df = df.withColumn("y6", df["y6"].cast(IntegerType()))
df = df.withColumn("y7", df["y7"].cast(IntegerType()))
df = df.withColumn("y8", df["y8"].cast(IntegerType()))
df = df.withColumn("y9", df["y9"].cast(IntegerType()))
df = df.withColumn("y10", df["y10"].cast(IntegerType()))
df = df.withColumn("y11", df["y11"].cast(IntegerType()))
df = df.withColumn("y12", df["y12"].cast(IntegerType()))
df = df.withColumn("y13", df["y13"].cast(IntegerType()))
df = df.withColumn("y14", df["y14"].cast(IntegerType()))
df = df.withColumn("y15", df["y15"].cast(IntegerType()))
df = df.withColumn("y16", df["y16"].cast(IntegerType()))
df = df.withColumn("y17", df["y17"].cast(IntegerType()))
df = df.withColumn("y18", df["y18"].cast(IntegerType()))
df = df.withColumn("y19", df["y19"].cast(IntegerType()))
df = df.withColumn("ds", df["ds"].cast(TimestampType()))

# COMMAND ----------

data = df.toPandas()
data.info()
data.head()

# COMMAND ----------

data.tail()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC 
# MAGIC ax = data.set_index('ds').plot(figsize=(12, 8))
# MAGIC ax.set_ylabel('Series')
# MAGIC ax.set_xlabel('Date')
# MAGIC 
# MAGIC plt.show()

# COMMAND ----------

data.drop(columns=['y','y1','y2','y3','y5','y6','y7','y8','y9','y10','y11','y12','y13','y14','y15','y16','y17','y18','y19'], inplace=True)
data.head()

# COMMAND ----------

#react>avid migration
data.rename(columns = {'y4':'y'}, inplace = True)
data.head()

# COMMAND ----------

data.drop(columns=['y1','y2','y3'], inplace=True)

# COMMAND ----------

#data.rename(columns = {'y1':'y'}, inplace = True)
data.head()

# COMMAND ----------

data.tail()

# COMMAND ----------

ax = data.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Series')
ax.set_xlabel('Date')

plt.show()

# COMMAND ----------

#new>avid migration plot
ax = data.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Series')
ax.set_xlabel('Date')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC SEASONAL DECOMPOSITION

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose

register_matplotlib_converters()
sns.set_style("darkgrid")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

data.set_index('ds',inplace=True)
data.index=pd.to_datetime(data.index)

# COMMAND ----------

plt.rc("figure",figsize=(12,6))

# COMMAND ----------

m = Prophet(
  growth = 'linear',
  changepoint_prior_scale = 0.001, 
  seasonality_prior_scale = 1.0
)
m.fit(data)
future = m.make_future_dataframe(periods=34, freq='Q', include_history=True)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast, xlabel='date', ylabel='metric')
display(forecast)
decompose_result = seasonal_decompose(data['y'], model='additive', period=4)
decompose_result.plot();

# COMMAND ----------

df1 = pd.merge(data, forecast, on='ds')
residuals = forecast['yhat'] - data['y']

# COMMAND ----------

residuals

# COMMAND ----------

residuals.plot();

# COMMAND ----------

# MAGIC %md
# MAGIC Migration series

# COMMAND ----------

m = Prophet(
  changepoints=['2020-03-31'],
  #changepoint_prior_scale = 0.01, 
  #seasonality_prior_scale = 0.01, 
  #seasonality_mode = 'multiplicative',
  growth = 'flat')
m.add_seasonality(name='quarterly', period=91.25, fourier_order=10, prior_scale=0.01)
m.fit(data)
future = m.make_future_dataframe(periods=34, freq='Q', include_history=True)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast, xlabel='date', ylabel='metric')
display(forecast)

# COMMAND ----------

#new>avid migration
m = Prophet(
  changepoints=['2020-03-31'], 
  changepoint_prior_scale = 0.001, 
  seasonality_prior_scale = 1.0, 
  seasonality_mode = 'multiplicative')
m.add_seasonality(name='quarterly', period=91.25, fourier_order=10, prior_scale=0.1)
m.fit(data)
future = m.make_future_dataframe(periods=34, freq='Q', include_history=True)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast, xlabel='date', ylabel='metric')
display(forecast)

# COMMAND ----------

m = Prophet(
  growth = 'linear',
  changepoint_prior_scale = 0.001, 
  seasonality_prior_scale = 1.0
)
m.fit(data)
future = m.make_future_dataframe(periods=34, freq='Q', include_history=True)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast, xlabel='date', ylabel='metric')
display(forecast)

# COMMAND ----------

fig2 = m.plot_components(forecast)

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

# COMMAND ----------

plot_components_plotly(m, forecast)

# COMMAND ----------

metric_df = forecast.set_index('ds')[['yhat']].join(data.set_index('ds').y).reset_index()
metric_df.tail()

# COMMAND ----------

metric_df.dropna(inplace=True)
metric_df.tail()

# COMMAND ----------

mean_squared_error(metric_df.y, metric_df.yhat)

# COMMAND ----------

mean_absolute_error(metric_df.y, metric_df.yhat)

# COMMAND ----------

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import itertools

# COMMAND ----------

data.head()

# COMMAND ----------

def create_param_combinations(**param_dict):
    param_iter = itertools.product(*param_dict.values())
    params =[]
    for param in param_iter:
        params.append(param) 
    params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
    return params_df

def single_cv_run(data, metrics, param_dict):
    m = Prophet(**param_dict)
    #m.add_country_holidays(country_name='US')
    m.fit(data)
    #df_cv = cross_validation(m, initial=34, period=4, horizon = 10)
    #df_p = performance_metrics(df_cv).mean().to_frame().T
    #df_p['params'] = str(param_dict)
    #df_p = df_p.loc[:, metrics]
    #return df_p

param_grid = {  
                'changepoint_prior_scale': [0.005, 0.05, 0.5, 5],
                'changepoint_range': [0.8, 0.9],
                'seasonality_prior_scale':[0.1, 1, 10.0],
                'holidays_prior_scale':[0.1, 1, 10.0],
                'seasonality_mode': ['multiplicative', 'additive'],
                #'growth': ['linear', 'logistic'],
                'yearly_seasonality': [5, 10, 20]
              }

metrics = ['horizon', 'rmse', 'mape', 'params'] 

results = []


params_df = create_param_combinations(**param_grid)
for param in params_df.values:
    param_dict = dict(zip(params_df.keys(), param))
    cv_df = single_cv_run(data,  metrics, param_dict)
    results.append(data)
results_df = pd.concat(results).reset_index(drop=True)
best_param = results_df.loc[results_df['mape'] == min(results_df['mape']), ['params']]
print(f'\n The best param combination is {best_param.values[0][0]}')
results_df

# COMMAND ----------

print(best_params)
print(results_df)

# COMMAND ----------

 param_grid = {  
'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
'seasonality_mode': ['multiplicative', 'additive'],
#'growth': ['linear', 'logistic'],
'yearly_seasonality':[5,10,20,40],
'weekly_seasonality':[5,10,20,40],
'daily_seasonality':[5,10,20,40],
}

# COMMAND ----------

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # store the RMSEs for each params here

# COMMAND ----------

cutoffs = pd.to_datetime(['2013-02-15', '2013-08-15', '2014-02-15'])
df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='365 days')

# COMMAND ----------

# use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(data)  # Fit model with given params
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='34', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# COMMAND ----------

tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

# COMMAND ----------

best_params = all_params[np.argmin(rmses)]
print(best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC Validation checks

# COMMAND ----------

# MAGIC %md
# MAGIC avid > elite migration parameter tuning 

# COMMAND ----------

end_date = '2019-06-30'
mask1 = (data['ds'] <= end_date)
mask2 = (data['ds'] > end_date)

# COMMAND ----------

X_tr = data.loc[mask1]
X_tst = data.loc[mask2]
print("train shape",X_tr.shape)
print("test shape",X_tst.shape)

# COMMAND ----------

pd.plotting.register_matplotlib_converters()
f, ax = plt.subplots(figsize=(14,5))
X_tr.plot(kind='line', x='ds', y='y', color='blue', label='Train', ax=ax)
X_tst.plot(kind='line', x='ds', y='y', color='red', label='Test', ax=ax)
plt.title('Series Traning and Test data')
plt.show()

# COMMAND ----------

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# COMMAND ----------

model =Prophet()
model.fit(X_tr)

# COMMAND ----------

future = model.make_future_dataframe(periods=12, freq='Q')
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

# COMMAND ----------

# Plot the components of the model
fig = model.plot_components(forecast)

# COMMAND ----------

f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(forecast,ax=ax)
plt.show()

# COMMAND ----------

X_tst_forecast = model.predict(X_tst)
X_tst_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

# COMMAND ----------

# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(X_tst.ds, X_tst['y'], color='r')
fig = model.plot(X_tst_forecast, ax=ax)

# COMMAND ----------

f, ax = plt.subplots(figsize=(14,5))
f.set_figheight(5)
f.set_figwidth(15)
X_tst.plot(kind='line',x='ds', y='y', color='red', label='Test', ax=ax)
X_tst_forecast.plot(kind='line',x='ds',y='yhat', color='green',label='Forecast', ax=ax)
plt.title('Forecast vs Actuals')
plt.show()

# COMMAND ----------

mape = mean_absolute_percentage_error(X_tst['y'],X_tst_forecast['yhat'])
print("MAPE",round(mape,4))

# COMMAND ----------

from sklearn.model_selection import ParameterGrid

params_grid = {'seasonality_mode':('multiplicative','additive'),
               'changepoint_prior_scale':[0.1,0.2,0.3,0.4,0.5],
              'holidays_prior_scale':[0.1,0.2,0.3,0.4,0.5],
              'n_changepoints' : [100,150,200]}
grid = ParameterGrid(params_grid)
cnt = 0
for p in grid:
    cnt = cnt+1

print('Total Possible Models',cnt)

# COMMAND ----------

import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

strt='2014-03-31'
end='2019-06-30'
model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
for p in grid:
    test = pd.DataFrame()
    print(p)
    random.seed(0)
    train_model = Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                         #holidays_prior_scale = p['holidays_prior_scale'],
                         n_changepoints = p['n_changepoints'],
                         seasonality_mode = p['seasonality_mode'],
                         #weekly_seasonality=False,
                         #daily_seasonality = False,
                         yearly_seasonality = True,
                         #holidays=holiday, 
                         interval_width=0.95)
    #train_model.add_country_holidays(country_name='US')
    train_model.fit(X_tr)
    train_forecast = train_model.make_future_dataframe(periods=12, freq='Q',include_history = False)
    train_forecast = train_model.predict(train_forecast)
    test=train_forecast[['ds','yhat']]
    Actual = data[(data['ds']>=strt) & (data['ds']<=end)]
    MAPE = mean_absolute_percentage_error(Actual['y'],abs(test['yhat']))
    print('Mean Absolute Percentage Error(MAPE)------------------------------------',MAPE)
    model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)

# COMMAND ----------

m = Prophet()
m.fit(data)
forecast = m.predict()
df1 = pd.merge(data, forecast, on='ds')
residuals = forecast['yhat'] - data['y']

# COMMAND ----------

residuals.plot
plt.show()

# COMMAND ----------

residuals.plot();

# COMMAND ----------

residuals
