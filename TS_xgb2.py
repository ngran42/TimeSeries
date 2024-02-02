# MAGIC %md
# MAGIC STEP ONE: LOAD LIBRARIES

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import xgboost as xgb
plt.style.use('seaborn-bright')

# COMMAND ----------

# MAGIC %md
# MAGIC STEP TWO: LOAD DATA

# COMMAND ----------

df = spark.read.format ("csv").load("dbfs:/mnt/customer-analytics/NGrannum/new_cust_xgb.csv", header=True)
df.head(5)

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType


df = df.withColumn("new_cust", df["new_cust"].cast(IntegerType()))
df = df.withColumn("quarter", df["quarter"].cast(TimestampType()))

# COMMAND ----------

from pyspark.sql.types import FloatType
from pyspark.sql.types import FloatType


df = df.withColumn("new_cust", df["new_cust"].cast(FloatType()))
df = df.withColumn("quarter", df["quarter"].cast(FloatType()))

# COMMAND ----------

df = df.toPandas()
df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

df.dtypes

# COMMAND ----------

print(df.shape)

# COMMAND ----------

df = df.set_index('quarter')
df.index = pd.to_datetime(df.index)

# COMMAND ----------

df['quarter'] = pd.to_datetime(df.quarter)

# COMMAND ----------

df.info()

# COMMAND ----------

df.head()

# COMMAND ----------

df.tail()

# COMMAND ----------

# check for data type
print(df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ADD FEATURE

# COMMAND ----------

df["target"] = df.new_cust.shift(-1)

# COMMAND ----------

df.dropna(inplace=True)

# COMMAND ----------

df.head(5)

# COMMAND ----------

df.to_numpy()

# COMMAND ----------

print(df.dtypes)

# COMMAND ----------

# check the number of instances
print(df.shape)

# COMMAND ----------

df.to_numpy()

# COMMAND ----------

data = df.to_numpy()

# COMMAND ----------

data

# COMMAND ----------

X, y = df.iloc[:,:-1],df.iloc[:,-1]

# COMMAND ----------

  df.info()

# COMMAND ----------

y, X = df.loc[:,'new_cust'].values, df.loc[:,['quarter']].values

# COMMAND ----------

data_dmatrix = xgb.DMatrix(data=X,label=y)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# COMMAND ----------

X_train.head()

# COMMAND ----------

y

# COMMAND ----------

X

# COMMAND ----------

df = pd.DataFrame(df, columns = ['target','new_cust'])
df['target'] = df['target'].apply(np.int64)
df['new_cust'] = df['new_cust'].apply(np.int64)
print(df)

# COMMAND ----------

def train_test_split(data, perc):
  data = data.values
  n = int(len(data) * (1 - perc))
  return data[:n], data[n:]

# COMMAND ----------

train, test = train_test_split(df, 0.2)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# COMMAND ----------

X_train

# COMMAND ----------

X = train[:, :-1]
y = train[:, -1]

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# COMMAND ----------

reg_mod = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.08,
    subsample=0.75,
    colsample_bytree=1, 
    max_depth=7,
    gamma=0,
)

# COMMAND ----------

scores = cross_val_score(reg_mod, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())

# COMMAND ----------

reg_mod.fit(X_train,y_train)

preds = reg_mod.predict(X_test)

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

# COMMAND ----------

plt.figure(figsize=(10, 5), dpi=80)
sns.lineplot(x='Year', y='Value', data=data)

# COMMAND ----------

plt.figure(figsize=(10, 5), dpi=80)
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="test")
plt.plot(x_ax, predictions, label="predicted")
plt.title("Carbon Dioxide Emissions - Test and Predicted data")
plt.legend()
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5), dpi=80)
df=pd.DataFrame(predictions, columns=['pred']) 
df['date'] = pd.date_range(start='8/1/2016', periods=len(df), freq='M')
sns.lineplot(x='date', y='pred', data=df)
plt.title("Carbon Dioxide Emissions - Forecast")
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
#import chart_studio
#import chart_studio.plotly as py 
#import chart_studio.tools

# COMMAND ----------

print(predictions)

# COMMAND ----------

result = pd.DataFrame(columns=['Time', 'Test', 'Predicted'])
result['Time'] = pd.date_range(start='9/30/1966', periods=653, freq='Q')
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
                   xaxis_title='Months',
                   yaxis_title='Mean Sunspot Values',
                   template='gridon')

fig.show()

# COMMAND ----------

df=pd.DataFrame(predictions, columns=['pred']) 
df['quarter'] = pd.date_range(start='2/28/2021', periods=len(df), freq='Q')
fig = px.line(df, x="date", y="pred")
# Edit the layout
fig.update_layout(title='Forecast',
                   xaxis_title='Date',
                   yaxis_title='Values',
                   template='gridon')
fig.show()

# COMMAND ----------



# COMMAND ----------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

# COMMAND ----------

from sklearn.metrics import r2_score
r2 = np.sqrt(r2_score(y_test, predictions))
print("R_Squared Score : %f" % (r2))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

