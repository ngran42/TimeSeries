# MAGIC %pip install chart_studio

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import chart_studio
import chart_studio.plotly as py 

# COMMAND ----------

df = spark.read.format ("csv").load("dbfs:/mnt/customer-analytics/foldername/new_cust_xgb.csv", header=True)
df.head(5)

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType


df = df.withColumn("new_cust", df["new_cust"].cast(IntegerType()))
df = df.withColumn("quarter", df["quarter"].cast(TimestampType()))

# COMMAND ----------

data = df.toPandas()

# COMMAND ----------

data.dtypes

# COMMAND ----------

#data['quarter'] = pd.to_datetime(data.quarter)

# COMMAND ----------

data['Year'] = pd.DatetimeIndex(data['quarter']).year
data['Month'] = pd.DatetimeIndex(data['quarter']).month

# COMMAND ----------

print(data.shape)

# COMMAND ----------

X, y =  data.loc[:,['Month', 'Year']].values, data.loc[:,'new_cust'].values
data_dmatrix = xgb.DMatrix(X,label=y)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# COMMAND ----------

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Tune parameters
# MAGIC 
# MAGIC max_depth: by giving a defined max depth of 12, the algorithm will not create more than 12 levels in each tree.
# MAGIC <br>
# MAGIC n_estimators: the number of trees in the model.
# MAGIC <br>
# MAGIC learning_rate: the learning speed of our algorithm. In our case, it is to equal 0.03. You can play around with this value until you reach the perfect rate.
# MAGIC <br>
# MAGIC subsample: the fraction of observations to be randomly sampled for each tree.
# MAGIC <br>
# MAGIC tree_method: allows you to choose the tree construction algorithm. Some other choices include hist, and approx.

# COMMAND ----------



# COMMAND ----------

model = GradientBoostingRegressor()
model.fit(X_train,y_train)

# COMMAND ----------

y_pred = model.predict(X_test)
print(r2_score(y_test,y_pred))

# COMMAND ----------

from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# COMMAND ----------

sns.distplot(y_test-y_pred)

# COMMAND ----------

sns.histplot(y_test-y_pred)

# COMMAND ----------



# COMMAND ----------

reg = XGBRegressor(n_estimators=1000, learning_rate=0.01)
reg.fit(X_train, 
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='mae')

# COMMAND ----------

predictions = reg.predict(X_test)

# COMMAND ----------

test_df = data[data['quarter'] >= '2021-03-31']
train_df = data[data['quarter'] < '2021-03-31']

# COMMAND ----------

test_df

# COMMAND ----------

test_df = test_df.reset_index().drop('index', axis=1)
test_df['predictions'] = pd.Series(predictions)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

plt.rcParams.update({'figure.figsize': (17, 3), 'figure.dpi':300})
fig, ax = plt.subplots()
sns.lineplot(data=data, x='quarter', y='new_cust')
sns.lineplot(data=test_df, x='quarter', y='predictions')
plt.grid(linestyle='-', linewidth=0.3)
ax.tick_params(axis='x', rotation=90)

# COMMAND ----------

train_df

# COMMAND ----------

print(r2_score(y_test,predictions))

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install darts

# COMMAND ----------

from darts import TimeSeries
from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries, sine_timeseries
from darts.models import RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel
from darts.metrics import mape, smape

# COMMAND ----------

model_new = NBEATSModel(input_chunk_length=34, output_chunk_length=34, n_epochs=100, random_state=0)

# COMMAND ----------

import pandas as pd

# COMMAND ----------

model_new.fit([data], verbose=True)

# COMMAND ----------

# Creating an empty Dataframe with column names only
result = pd.DataFrame(columns=['Time', 'Test', 'Predicted'])
result['Time'] = pd.date_range(start='03/31/2021', periods=6, freq='Q')
result['Test'] = y_test
result['Predicted'] = predictions
#Using Plotly to build the graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=result['Time'], y=result['Test'],
                    mode='lines',
                    name='Test'))
fig.add_trace(go.Scatter(x=result['Time'], y=result['Predicted'],
                    mode='lines',
                    name='Predicted'))

# Edit the layout
fig.update_layout(title='Test vs Predicted Values',
                   xaxis_title='Quarter',
                   yaxis_title='New Cust Cnt',
                   template='gridon')

fig.show()

# COMMAND ----------

sns.distplot(y_test-y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC Tune model

# COMMAND ----------

from xgboost import XGBRegressor
from numpy import nan, log

reg_mod = xgb.XGBRegressor(objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.10,
    subsample=0.5,
    colsample_bytree=1, 
    max_depth=5, 
)
reg_mod.fit(X_train, y_train)
XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# COMMAND ----------

reg_mod.fit(X_train,y_train)

predictions = reg_mod.predict(X_test)

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

# COMMAND ----------

from xgboost import XGBRegressor
from numpy import nan, log

reg_mod = xgb.XGBRegressor(objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.10,
    subsample=0.5,
    colsample_bytree=1, 
    max_depth=5, 
)
reg_mod.fit(X_train, y_train)
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=1000, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.5,
             tree_method='exact', validate_parameters=1, verbosity=None)

# COMMAND ----------

reg_mod.fit(X_train,y_train)

predictions = reg_mod.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE (Root Mean Squared Error) of an estimator of a population parameter is the square root of the mean square error (MSE). The mean square error is defined as the expected value of the square of the difference between the estimator and the parameter. It is the sum of variance and squared bias.
# MAGIC 
# MAGIC R-squared value, denoted by R2 is a statistical measure that calculates the proportion of variation in the dependent variable that can be attributed to the independent variable.

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

# COMMAND ----------

from sklearn.metrics import r2_score
r2 = np.sqrt(r2_score(y_test, predictions))
print("R_Squared Score : %f" % (r2))

# COMMAND ----------

# Creating an empty Dataframe with column names only
result = pd.DataFrame(columns=['Time', 'Test', 'Predicted'])
result['Time'] = pd.date_range(start='03/31/2021', periods=6, freq='Q')
result['Test'] = y_test
result['Predicted'] = predictions
#Using Plotly to build the graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=result['Time'], y=result['Test'],
                    mode='lines',
                    name='Test'))
fig.add_trace(go.Scatter(x=result['Time'], y=result['Predicted'],
                    mode='lines',
                    name='Predicted'))

# Edit the layout
fig.update_layout(title='Test vs Predicted Values',
                   xaxis_title='Quarter',
                   yaxis_title='New Cust Cnt',
                   template='gridon')

fig.show()

# COMMAND ----------

df=pd.DataFrame(predictions, columns=['pred']) 
df['date'] = pd.date_range(start='9/30/2022', periods=len(df), freq='Q')
fig = px.line(df, x="date", y="pred")
# Edit the layout
fig.update_layout(title='New Cust Cnt Forecast',
                   xaxis_title='Date',
                   yaxis_title='Counts',
                   template='gridon')
fig.show()

# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC EXPLORE ROLLING FORECAST
# MAGIC 
# MAGIC To reduce this error and avoid the bias we can do rolling forecast, in which we will use use the latest prediction value in the forecast for next time period. This can be done by re-creating SARIMA model after each observation received. We will manually keep track of all observations in a list called history that is seeded with the training data and to which new observations are appended each iteration.

# COMMAND ----------

data.new_cust.rolling(3, win_type ='triang').sum()

# COMMAND ----------

data.new_cust.rolling(3).mean()

# COMMAND ----------

