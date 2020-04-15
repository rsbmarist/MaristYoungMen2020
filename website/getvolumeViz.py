import os
from matplotlib import pyplot as plt
import csv
import pandas as pd
from datetime import datetime, timedelta
# initalize wd to visualization folder
os.chdir(r'C:\Users\Christopher Major\OneDrive - Marist College\MaristYoungMen2020\visualization')
print(os.getcwd())


def GetVolume(fname, days='all'):
    # set wd to data\DailyValues
    os.chdir(r'C:\Users\Christopher Major\OneDrive - Marist College\MaristYoungMen2020\data\DailyValues')
    symbol = fname[:-9]
    df = pd.read_csv(fname)
    df['Date'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) # save time string as datetime
    
    if days != 'all':
        # make functions for last week, month, year
        x=1
    df[symbol]=df['volume']
    ax = df.plot(x='Date',y=symbol, title=symbol+' Performance Over Time', figsize=(10,8))
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Volume')
    os.chdir(r'C:\Users\Christopher Major\OneDrive - Marist College\MaristYoungMen2020\visualization\WebsitePNGs')
    plt.savefig(symbol+'VolumeAllTime.png', bbox_inches='tight', dpi=350)
    print('saved '+symbol+'VolumeAllTime.png'+' to '+os.getcwd())

stockSymbol = ["AAPL", "AMZN", "GOOGL","MSFT", "DELL", "IBM", "INTC", "HPQ",
               "FB", "CSCO", "ORCL", "HPE", "MU", "DXC", "TMO"]
for i in stockSymbol:
    GetVolume('{}Daily.csv'.format(i))