import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from download_date import import_data


pd.options.mode.chained_assignment = None
# Data download
df = import_data()

# Create a variable to predict 'x' days out into the future
future_days = 30


# Draw line plot
def draw_line_chart():
    plt.subplots(figsize=(16, 6))
    plt.plot(df.index, df['Close'], color='blue', linewidth=1)

    plt.title('Historical Currency Exchange Rate PLN/USD')

    plt.xlabel('Date')
    plt.ylabel('Price [PLN]')
    plt.grid()
    plt.show()


# Draw heatmap
def draw_heatmap():
    print(df.corr())
    sns.heatmap(df.corr())
    plt.show()


def data_creation():
    # Create a new column (target) shifted 'x' units/days up
    df['Prediction'] = df[['Close']].shift(-future_days)
    return df


def future_data_x():
    # Create feature data set (X) and convert it to a numpy array and remove the last 'x' rows/days
    X = np.array(df.drop(columns='Prediction'))[:-future_days]
    return X


def future_data_y():
    # Create the target date set (y) and convert it to a numpy array and get all of the target values except the last 'x' rows
    y = np.array(df['Prediction'])[:-future_days]
    print(y)
    return y


def last_x_future():
    # Get the last 'x' rows of the feature data set
    x_future = data_creation().drop(columns='Prediction')[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    return x_future


def model_creation_dtr():
    # Split the data into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(future_data_x(), future_data_y(), test_size=0.25)

    # Create the decision tree regressor model
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    # Show the model tree prediction
    tree_prediction = tree.predict(last_x_future())
    print(tree_prediction)
    return tree_prediction


def visualize_data_tree_prediction():
    # Visualize the date tree prediction
    predictions = model_creation_dtr()
    valid = df[future_data_x().shape[0]:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model tree prediction')
    plt.xlabel('Date')
    plt.ylabel('Price [PLN]')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Oryginal data', 'Valid data', 'Predicted data'])
    plt.grid()
    plt.show()


def model_creator_linear_regression():
    # Split the data into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(future_data_x(), future_data_y(), test_size=0.25)

    # Create the linear regression model
    lr = LinearRegression().fit(x_train, y_train)
    # Show the model linear regression prediction
    lr_prediction = lr.predict(last_x_future())
    print(lr_prediction)
    return lr_prediction


def visualize_data_linear_regression_prediction():
    # Visualize the date linear regression prediction
    predictions = model_creator_linear_regression()
    valid = df[future_data_x().shape[0]:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model linear regression prediction')
    plt.xlabel('Date')
    plt.ylabel('Price [PLN]')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Oryginal data', 'Valid data', 'Predicted data'])
    plt.grid()
    plt.show()
