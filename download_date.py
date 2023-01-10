import yfinance as yf
import datetime as dt


# Download stock data then export to CSV
def import_data():
    df_yahoo = yf.download("PLN=X", start=dt.date.today()-dt.timedelta(days=3*365), end=dt.date.today(), progress=False)
    return df_yahoo
