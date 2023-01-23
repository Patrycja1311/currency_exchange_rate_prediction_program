import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from download_date import import_data
import datetime as dt


pd.options.mode.chained_assignment = None
# Data download
df = import_data()


# Create a variable to predict 'x' days out into the future
future_days = 30


# Draw line plot
def draw_line_chart():
    plt.subplots(figsize=(14, 7))
    plt.plot(df.index, df['Close'], color='blue', linewidth=1)
    plt.title('Historical Currency Exchange Rate PLN/USD')
    plt.xlabel('Date')
    plt.ylabel('Price [PLN]')
    plt.grid()
    plt.show()


# Draw heatmap
def draw_heatmap():
    sns.heatmap(df.corr())
    plt.show()


def future_data_x():
    x = df[['Open', 'High', 'Low']]
    x = x.to_numpy()
    return x


def future_data_y():
    y = df['Close']
    y = y.to_numpy()
    y = y.reshape(-1, 1)
    return y


def model_creation_dtr():
    x_train, x_test, y_train, y_test = train_test_split(future_data_x(), future_data_y(), test_size=0.25)
    tree = DecisionTreeRegressor()
    tree.fit(x_train, y_train)
    tree_prediction = tree.predict(x_test)
    return tree_prediction


def prediction_date():
    predictions = pd.DataFrame({'Predict Rate': model_creation_dtr().flatten()})
    predictions = predictions.head(future_days)
    df['Predictions'] = predictions
    df_new = pd.DataFrame()
    df_new['Date'] = pd.date_range(start=dt.date.today(), end=dt.date.today() + dt.timedelta(days=future_days))
    df_new['Predictions'] = predictions
    return df_new


def visualize_data_tree_prediction():
    plt.figure(figsize=(14, 7))
    plt.title('Model tree prediction')
    plt.xlabel('Date')
    plt.ylabel('Price [PLN]')
    plt.plot(df['Close'], label='Oryginal')
    plt.plot(prediction_date()['Date'], prediction_date()['Predictions'], label='Prediction')
    plt.legend()
    plt.grid()
    plt.show()


def model_creation_linear_regression():
    x_train, x_test, y_train, y_test = train_test_split(future_data_x(), future_data_y(), test_size=0.25)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_prediction = lr.predict(x_test)
    return lr_prediction


def prediction_date_lr():
    predictions = pd.DataFrame({'Predict Rate': model_creation_linear_regression().flatten()})
    predictions = predictions.head(future_days)
    df['Predictions'] = predictions
    df_new = pd.DataFrame()
    df_new['Date'] = pd.date_range(start=dt.date.today(), end=dt.date.today() + dt.timedelta(days=future_days))
    df_new['Predictions'] = predictions
    return df_new


def visualize_data_linear_regression_prediction():
    plt.figure(figsize=(14, 7))
    plt.title('Model linear regression prediction')
    plt.xlabel('Date')
    plt.ylabel('Price [PLN]')
    plt.plot(df['Close'], label='Oryginal')
    plt.plot(prediction_date_lr()['Date'], prediction_date_lr()['Predictions'], label='Prediction')
    plt.legend()
    plt.grid()
    plt.show()
