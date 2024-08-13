import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# model
import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

#chage the directory to desired file
df = pd.read_csv("./DataSet/DataSetForModel/UsableDataSet/ProjectDataSetFor5Years.csv")

df = df.set_index("date")
df.index = pd.to_datetime(df.index)
df.plot(style = ".",
        figsize= (15, 5),
        color = color_pal[0],
        title = "Project Work Traffic")
plt.show()

# Train/Test Split
train = df.loc[df.index < "01-01-2022"]
test = df.loc[df.index >= "01-01-2022"]

fig, ax = plt.subplots(figsize = (15, 5))
train.plot(ax = ax, label = "Training Set", title = "Date Train/Test Split")
test.plot(ax= ax, label = "Test Set")
ax.axvline("01-01-2022", color = "black", ls = "--")
ax.legend(["Training Set", "Test Set"])
plt.show()

df.loc[(df.index > "14-02-2019") & (df.index < "21-02-2019")].plot(figsize = (15, 5), title = "Week of date")
plt.show()

df.index.dayofyear

def create_feature(df):
    """
    Create time series features based on time series index
    """
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["day"] = df.index.day
    return df
df = create_feature(df)

# visualaize our feature / Target Relationship
fig, ax = plt.subplots(figsize = (10, 8))
sns.boxplot(data = df, x = "month", y = 'INSTANCE')
ax.set_title("WorkTraffic by Month")
plt.show

train = create_feature(train)
test = create_feature(test)

FEATURES = ["quarter", "month", "year", "day"]
TARGET = "INSTANCE"

x_train = train[FEATURES]
y_train = train[TARGET]

x_test = test[FEATURES]
y_test = test[TARGET]

# create aour model
# regression model
reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50,
                      learning_rate = 0.001)
reg.fit(x_train, y_train,
    eval_set = [(x_train, y_train), (x_test, y_test)],
    verbose = 100)

# forecast on Test
test["prediction"] = reg.predict(x_test)
df = df.merge(test[["prediction"]], how = "left", left_index = True, right_index = True)
ax = df[['INSTANCE']].plot(figsize = (15, 5))
df["prediction"].plot(ax=ax, style = "-")
plt.legend(["Truth Data", "prediction"])
ax.set_title("Raw Data and Prediction")
plt.show()