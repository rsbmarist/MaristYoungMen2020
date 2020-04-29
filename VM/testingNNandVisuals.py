# ## Make predictions



#Import statements
import json
import requests
import pandas as pd
import numpy as np
import datetime
from datetime import date
from pandas.io.json import json_normalize
from pathlib import Path
import pytz
from datetime import timedelta
import os
import os.path
from os import path
import csv
import time
import re
import string
from dateutil import tz
import demoji
from collections import Counter
import ast
demoji.download_codes()


# In[143]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score 
from sklearn.metrics import confusion_matrix,roc_curve, auc


# In[144]:


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Dropout

stockSymbol = ["AAPL", "AMZN", "GOOGL", "MSFT", "DELL", "IBM", "INTC", "HPQ", "FB",
	 "CSCO", "ORCL", "HPE", "MU", "DXC", "TMO"]


# In[145]:


#from_zone = tz.gettz('UTC')
#to_zone = tz.gettz('America/New_York')

# This is the current list the NYSE will be shut down to observe holidays
holidayList = [datetime.date(2020, 4, 10), datetime.date(2020, 5, 25), datetime.date(2020, 7, 3), datetime.date(2020, 9, 7), datetime.date(2020, 11, 26), datetime.date(2020, 12, 25),
              datetime.date(2021, 1, 1), datetime.date(2021, 1, 18), datetime.date(2021, 2, 15), datetime.date(2021, 4, 2), datetime.date(2021, 5, 31), datetime.date(2021, 7, 5),
              datetime.date(2021, 9, 6), datetime.date(2021, 11, 25), datetime.date(2021, 12, 24), datetime.date(2022, 1, 17), datetime.date(2022, 2, 21), datetime.date(2022, 4, 15),
              datetime.date(2022, 5, 30), datetime.date(2022, 7, 4), datetime.date(2022, 9, 5), datetime.date(2022, 11, 24), datetime.date(2022, 12, 26)]



# In[152]:


def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()


# In[153]:


def start_predictions():
    #re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


    fullDf = pd.DataFrame(columns=['id', 'cleanSents', 'tag', 'created'])
    currentFolder = os.getcwd()

    #Will need to switch file path to / instead of \\ for deployment 
    for symbol in stockSymbol:
        df = pd.read_csv(currentFolder+"/{}folder/{}_twits.csv".format(symbol, symbol))[{'id', 'cleanSents', 'tag', 'created'}]
        fullDf = fullDf.append(df)
    fullDf = fullDf.drop_duplicates('id')
    fullDf = fullDf.reset_index()[{'id','cleanSents','tag', 'created'}]

    bullish = fullDf[fullDf['tag'] == 'Bullish'].reset_index()[{'id','cleanSents','tag', 'created'}]
    bearish = fullDf[fullDf['tag'] == 'Bearish'].reset_index()[{'id','cleanSents','tag', 'created'}]
    none = fullDf[fullDf['tag'] == 'none'].reset_index()[{'id','cleanSents','tag', 'created'}]
    taggedDf = bullish.append(bearish)

    X_train, X_test, y_train, y_test = train_test_split(taggedDf['cleanSents'].values, taggedDf['tag'].values, test_size=0.2)
    vect = TfidfVectorizer()

    tf_train = vect.fit_transform(X_train)
    tf_test = vect.transform(X_test)

    NLPmodel = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', class_weight='balanced').fit(tf_train, y_train)

    predictions = NLPmodel.predict(tf_test)

    # Report the predctive performance metrics
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    if not (path.exists(currentFolder+"nlpDailyPerformance.csv")):
        NLPdailyPerformance = pd.DataFrame(columns = ["Accuracy", "Balanced_Accuracy", "Date"])
        NLPdailyPerformance = NLPdailyPerformance.append({"Accuracy" : accuracy,
                                          "Balanced_Accuracy" : balanced_accuracy,
                                            "Date" : datetime.datetime.today().date()}, ignore_index=True)
    else:
        NLPdailyPerformance = pd.read_csv("nlpDailyPerformance.csv")
        NLPdailyPerformance = NLPdailyPerformance.append({"Accuracy" : accuracy,
                                          "Balanced_Accuracy" : balanced_accuracy,
                                            "Date" : datetime.datetime.today().date()}, ignore_index=True)
    NLPdailyPerformance.to_csv("nlpDailyPerformance.csv", index=False)
    make_datasets(NLPmodel, vect)


# In[154]:


def count_predict(df, NLPmodel, vect):
    if len(df) == 0:
        return 0
    tf_new = vect.transform(df['cleanSents'])
    # get probabilities for positive class
    probs = NLPmodel.predict_proba(tf_new)[:,1]
    preds = NLPmodel.predict(tf_new)
    return len(preds[preds=='Bullish'])


# In[155]:


def make_prediction(company, companyDf, newDataDf, lastClose):
    currentFolder = os.getcwd()
    #Setup Neural Network
    #Variables
    x=companyDf[{'prePct_traded_vol', 'prePct_close_val', 'percent_bullish', 'pct_twits_volume'}]
    y=companyDf['percentChange']
    y=np.reshape(y.values, (-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    xscale=scaler_x.transform(x)
    yscale=scaler_y.transform(y)
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=.30)
    
    #Build Neural Network
    NNmodel = Sequential()
    NNmodel.add(Dense(50, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    NNmodel.add(Dropout(0.5))
    NNmodel.add(Dense(100, activation='relu'))
    NNmodel.add(Dropout(0.2))
    NNmodel.add(Dense(10, activation='relu'))
    NNmodel.add(Dropout(0.5))
    NNmodel.add(Dense(1, activation='relu'))
    NNmodel.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    
    #Recall the best validation loss and reduced learning rate and stop training
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20,  
                              min_delta=1e-4, mode='min')

    stop_alg = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    history = NNmodel.fit(X_train, y_train, batch_size=50,  verbose=0, validation_split=0.2, epochs=1000, 
                  callbacks=[stop_alg, reduce_lr])
    
    #Used to see the loss of both validation and training over the course of training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(currentFolder+"/{}folder/{}nnLoss.png".format(company, company))
    
    plt.close()
    
    #Absolute error historgram
    pred = scaler_y.inverse_transform(NNmodel.predict(X_test))
    actual = scaler_y.inverse_transform(y_test)
    #AvgError = plt.hist(np.abs(actual-pred), bins=11)
    #AvgError.savefig(currentFolder+"/{}folder/{}nnAE.png".format(company, company))
    
    meanAbsError = np.mean(np.abs(actual-pred))
    meanError = np.mean(actual-pred)
    stdError = np.std(actual-pred)
    
    if (path.exists(currentFolder+"/{}folder/{}nnDailyPerformance.csv".format(company, company))):
        NNdailyPerformance = pd.read_csv(currentFolder+"/{}folder/{}nnDailyPerformance.csv".format(company, company))
        NNdailyPerformance = NNdailyPerformance.append({"MeanAbsError" : meanAbsError,
                                                      "MeanError" : meanError,
                                                      "StdError" : stdError,
                                                      "Date" : datetime.datetime.today().date()}, ignore_index=True)
    
    else:
        NNdailyPerformance = pd.DataFrame(columns = ["MeanAbsError", "MeanError", "StdError", "Date"])
        NNdailyPerformance = NNdailyPerformance.append({"MeanAbsError" : meanAbsError,
                                              "MeanError" : meanError,
                                              "StdError" : stdError,
                                              "Date" : datetime.datetime.today().date()}, ignore_index=True)
        
    NNdailyPerformance.to_csv(currentFolder+"/{}folder/{}nnDailyPerformance.csv".format(company, company), index=False)
    
    #Compute the newest prediction
    
    newData = scaler_y.transform(newDataDf)
    newPrediction = scaler_y.inverse_transform(NNmodel.predict(newData))[0]
    
    
    
    diff = lastClose*newPrediction
    newClosePred = lastClose+diff
    
    #df = pd.DataFrame([newPrediction], columns=['prediction'], 
    #              index=['{}'.format(company)])
    #df['positive'] = df['prediction']>0
    
    df = pd.DataFrame([{'lastClose' : lastClose, 'prediction': round(newClosePred[0], 2)}], 
                  index=['{} Last Close'.format(company), '{} Prediciton'.format(company)])
    df['positive'] = ((df['prediction']-df['lastClose'])>0)
    
    columns = ('Last Close', 'Prediciton')
    y_pos = np.arange(len(columns))
    values = [lastClose, round(newClosePred[0], 2)]
    
    x=np.arange(len(columns))
    width = .8
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, values, width, color =df.positive.map({True: 'g', False:'r'}))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Stock Value')
    ax.set_title('{} Stock Prediction for {}'.format(company, date.today()))
    ax.set_xticks(x)
    ax.set_xticklabels(columns)

    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -25),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                   fontsize = 20) 
    
    #plt.bar(y_pos, values, align='center', alpha=0.5, color=df.positive.map({True: 'g', False:'r'}))
    #plt.xticks(y_pos, columns)
    
    #plt.text(, lastClose*.5, str(lastClose))
    #plt.text(1, newClosePred*.5, str(newClosePred))

    #ax = df.plot(kind='bar', color=df.positive.map({True:'g', False:'r'}), alpha=.65) 
    #x_offset = -0.5
    #y_offset = 0.02
    #for p in ax.patches:
    #    ax.annotate(str(p.get_height()), (p.get_x()*0.6, p.get_height() * .5), fontsize=20)

    plt.savefig(currentFolder+"/{}folder/{}nnPrediction.png".format(company, company))
    plt.close()
        
    #print(newPrediction)
    


# In[178]:


#This function makes the correct dataframes inorder to make a prediction
def make_datasets(nlpmodel, vect):
    
    currentFolder = os.getcwd()

    est = pytz.timezone('US/Eastern')

    for symbol in stockSymbol:

        #Format twits data to prefered times and types.
        dfTwits = pd.read_csv(currentFolder+"/{}folder/{}_twits.csv".format(symbol, symbol))
        dfTwits['created'] = pd.to_datetime(dfTwits['created'])

        estList = []
        for index, row in dfTwits.iterrows():
            estList.append(est.localize(dfTwits['created'][index].to_pydatetime()))
        dfTwits['estTime'] = estList

        #Format daily stock market values to preferred 
        dfDaily = pd.read_csv(currentFolder+"/{}folder/{}Daily.csv".format(symbol, symbol))
        dfDaily['time'] = pd.to_datetime(dfDaily['time'])

        lowest = '2019-12-20'
        twits_vol = []
        pct_bullish = []
        preModeldf = dfDaily[dfDaily['time']>=lowest]
        preModeldf = preModeldf.sort_values(by='time')
        preModeldf = preModeldf.set_index('time')
        
        lastClose = preModeldf['close'][-1]

        preModeldf['prePct_traded_vol'] = preModeldf['percentVol'].shift(1)
        preModeldf['prePct_close_val'] = preModeldf['percentChange'].shift(1)
        preModeldf = preModeldf[{'prePct_traded_vol', 'prePct_close_val', 'percentChange'}]


        for row in preModeldf.index[:]:
            upper = row.to_pydatetime().replace(tzinfo= est) + timedelta(hours = 8, minutes = 30)
            lower = row.to_pydatetime().replace(tzinfo = est) + timedelta(days = -1, hours = 8, minutes = 30)
            windowTwits = dfTwits[(dfTwits['estTime']>=lower) & (dfTwits['estTime']<=upper)]
            volume = len(windowTwits)
            twits_vol.append(volume)

            if(volume<1):
                pct_bullish.append(.5)
                volume = 1
            else:
                bullish_count = len(windowTwits[windowTwits['tag']=='Bullish']) + count_predict(windowTwits[windowTwits['tag']=='none'],
                                                                                               nlpmodel,
                                                                                               vect)
                pct_bullish.append(bullish_count/volume)
        preModeldf['percent_bullish'] = pct_bullish
        preModeldf['twits_volume'] = twits_vol
        modeldf = preModeldf
        modeldf['pct_twits_volume'] = modeldf['twits_volume'].pct_change(1)
        modeldf = modeldf[{'percentChange', 'prePct_traded_vol', 'prePct_close_val', 'percent_bullish', 'pct_twits_volume'}]
        modeldf=modeldf.replace([np.inf, -np.inf], 0.0).dropna()

        
        #gather newest row for predictions makes a separate dataframe
        predictRow = pd.DataFrame( columns = ['prePct_traded_vol', 'prePct_close_val', 'percent_bullish', 'pct_twits_volume'])
        dt = date.today()
        newUpper= datetime.datetime.combine(dt, datetime.datetime.min.time()).replace(tzinfo=est) + timedelta(hours=8, minutes=30)
        newLower=datetime.datetime.combine(dt, datetime.datetime.min.time()).replace(tzinfo=est) + timedelta(days=-1, hours=8, minutes=30)
        newWindowTwits = dfTwits[(dfTwits['estTime']>=newLower) & (dfTwits['estTime']<=newUpper)]
        newVolume = len(newWindowTwits)
        newBullish = .5
        if (newVolume <1):
            newBullish = .5
            newVolume = 1
        else:
            new_bullish_count = len(newWindowTwits[newWindowTwits['tag']=='Bullish']) + count_predict(newWindowTwits[newWindowTwits['tag']=='none'],
                                                                                                     nlpmodel,
                                                                                                     vect)
            newBullish = new_bullish_count/newVolume
        previousPctTradeVol = dfDaily[0:1]["percentVol"][0]
        previousPctTradeChange = dfDaily[0:1]["percentChange"][0]
        percentTwitsVolume = (newVolume - volume)/volume

        predictRow = predictRow.append(
            {'prePct_traded_vol':previousPctTradeVol,
             'prePct_close_val':previousPctTradeChange,
             'percent_bullish':newBullish,
             'pct_twits_volume':percentTwitsVolume}, ignore_index=True)
        
        #Make predicitons for the current day
        make_prediction(symbol, modeldf, predictRow, lastClose)


# ## Visualizations

# In[186]:


def MakeDonutChart(data,symbol,timeSeries, script, images):
    print(timeSeries)
    getCount = Counter(k['symbol'] for k in data if dict(k).get('symbol'))
    symbolCount = dict(getCount)
    symbolCount[symbol] = 0
    symbolCount = {k: v for k, v in sorted(symbolCount.items(), key=lambda item: item[1])}
    fig, ax = plt.subplots(figsize=(11, 10), subplot_kw=dict(aspect="equal"))

    cnt = 0
    data = []
    symbols = []
    recipe = []

    for sign, count in getCount.most_common():
        if cnt >= 1:
            data.append(count)
            symbols.append(sign)
            recipe.append(sign +' - '+str(count)+' twits')
        cnt += 1
        if cnt >= 6:
            break


#    for key in symbolCount.keys():
#        data.append(symbolCount[key])
#        symbols.append(key)
#        recipe.append(key + ' - ' + str(symbolCount[key]) + ' twits')
#        cnt += 1
#        if cnt >= 5:
#            break

    if sum(data) < 15:
        print('Not enough data for given time frame')
        fig.suptitle('Not enough data for given time frame', fontsize=20)
        os.chdir(images)
        plt.savefig(symbol+'TopFiveOtherCompanies'+timeSeries+'.png', optimize=True)
        #print('saved '+symbol+'TopFiveOtherCompanies'+timeSeries+'.png to '+os.getcwd())
        plt.close()
        os.chdir(script)
        #print('returning to '+os.getcwd())
        return

    def explode():
        try:
            exp = (0.1,0,0,0,0)
        except:
            exp=None
        return(exp)

    wedges, texts = ax.pie(data, 
                           explode=explode(), 
                           shadow=True, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.legend(wedges, symbols,
            fontsize='large',
            title_fontsize='large',
            title="Symbols",
            loc="center",
            frameon=False)
            #bbox_to_anchor=(1, 0, 0.5, 1))
    #ax.set_title("Top Ten Companies Mentioned in " + symbol + " Twits", fontsize=30, pad=50)

    os.chdir(images)
    plt.savefig(symbol+'TopFiveOtherCompanies'+timeSeries+'.png', optimize=True)
    #print('saved '+symbol+'TopFiveOtherCompanies'+timeSeries+'.png to '+os.getcwd())
    plt.close()
    os.chdir(script)
    #print('returning to '+os.getcwd())


# In[187]:


def GetOtherCompanies(fname, days, script, images):
    to_ignore = fname[:-10]
    # set wd to Symbolfolder
    symbolFolder = os.path.join(script, to_ignore+'folder')
    os.chdir(symbolFolder)
    #print('Pulling data from ' +os.getcwd())
    print(days)
    df = pd.read_csv(fname)
    df['Date'] = pd.to_datetime(df['created']) # save time string as datetime
    stock_ds = []
    if days == 'lastWeek':
        lastweekdate = pd.to_datetime('today').floor('D') - timedelta(7)
        df = df[df.Date >= lastweekdate]
        for row in df.newSymbols:
            lists = ast.literal_eval(row)
            for diction in lists:
                stock_ds.append(diction)
        MakeDonutChart(stock_ds, to_ignore, 'LastWeek', script, images)
    elif days == 'lastMonth':
        lastmonthdate = pd.to_datetime('today').floor('D') - timedelta(30)
        df = df[df.Date >= lastmonthdate]
        for row in df.newSymbols:
            lists = ast.literal_eval(row)
            for diction in lists:
                stock_ds.append(diction)
        print(len(stock_ds))
        MakeDonutChart(stock_ds, to_ignore, 'LastMonth', script, images)
    elif days == 'lastYear':
        lastyeardate = pd.to_datetime('today').floor('D') - timedelta(365)
        df = df[df.Date >= lastyeardate]
        for row in df.newSymbols:
            lists = ast.literal_eval(row)
            for diction in lists:
                stock_ds.append(diction)
        print(len(stock_ds))
        MakeDonutChart(stock_ds, to_ignore, 'LastYear', script, images)
    else:
        for row in df.newSymbols:
            lists = ast.literal_eval(row)
            for diction in lists:
                stock_ds.append(diction)
        MakeDonutChart(stock_ds, to_ignore, 'AllTime', script, images)


# In[188]:


def MakeBarChart(tags,symbol,timeSeries, script, images):
    data = []
    tag = []
    for key in tags.keys():
        data.append(tags[key])
        tag.append(key)
    
    df = pd.DataFrame({'Tags':tag, 'val':data})
    ax = df.plot.barh('Tags', 'val', color=['y', 'r', 'g'], fontsize=15, figsize=(11,10), legend=False)
    ax.set_ylabel('Tags', fontsize=20)
    for i, v in enumerate(data):
        ax.text(v, i, str(v), fontsize=15, fontweight='bold')
    
    os.chdir(images)
    plt.savefig(symbol+'Tags'+timeSeries+'.png', optimize=True)
    #print('saved '+symbol+'Tags'+timeSeries+'.png to '+os.getcwd())
    plt.close()
    os.chdir(script)
    #print('returning to '+os.getcwd())


# In[189]:


def GetTags(fname, days, script, images):
    
    symbol = fname[:-10]
    # set wd to Symbolfolder
    symbolFolder = os.path.join(script, symbol+'folder')
    os.chdir(symbolFolder)
    #print('Pulling data from ' +os.getcwd())
    df = pd.read_csv(fname)
    df['Date'] = pd.to_datetime(df['created']) # save time string as datetime
    tags = {
        'Bullish': 0,
        'Bearish': 0,
        'none': 0
    }

    if days == 'lastWeek':
        lastweekdate = pd.to_datetime('today').floor('D') - timedelta(7)
        df = df[df.Date >= lastweekdate]
        for row in df.tag:
            tags[row] += 1
        MakeBarChart(tags,symbol,'LastWeek', script, images)
    elif days == 'lastMonth':
        lastmonthdate = pd.to_datetime('today').floor('D') - timedelta(30)
        df = df[df.Date >= lastmonthdate]
        for row in df.tag:
            tags[row] += 1
        MakeBarChart(tags,symbol,'LastMonth', script, images)
    elif days == 'lastYear':
        lastyeardate = pd.to_datetime('today').floor('D') - timedelta(365)
        df = df[df.Date >= lastyeardate]
        for row in df.tag:
            tags[row] += 1
        MakeBarChart(tags,symbol,'LastYear', script, images)
    else:
        for row in df.tag:
            tags[row] += 1
        MakeBarChart(tags,symbol,'AllTime', script, images)


# In[190]:


def GetVolume(fname, days, script, images):
    
    if 'Values' in fname:
        symbol = fname[:-10]
    else:
        symbol = fname[:-9]
    symbolFolder = os.path.join(script, symbol+'folder')
    os.chdir(symbolFolder)
    #print('Pulling data from ' +os.getcwd())
    df = pd.read_csv(fname)
    df['Date'] = pd.to_datetime(df['time'])
    df["SMA1"] = df['close'].rolling(window=25).mean()
    df["SMA2"] = df['close'].rolling(window=100).mean()
    
    if days == 'lastWeek':
        lastweekdate = pd.to_datetime('today').floor('D') - timedelta(7)
        df = df[df.Date >= lastweekdate]
        ax = df.plot('Date', 'volume', figsize=(15,10), fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel('Volume', fontsize=20)
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'VolumeLastWeek.png', optimize=True)
        #print('saved '+symbol+'VolumeLastWeek.png to '+os.getcwd())
        plt.close()
        os.chdir(symbolFolder)
        #print('returning to '+os.getcwd())
        ax = df.plot('Date', 'close', figsize=(11,10), fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel('Price', fontsize=20)
        plt.legend()
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'PriceLastWeek.png', optimize=True)
        #print('saved '+symbol+'PriceLastWeek.png to '+os.getcwd())
        plt.close()
        os.chdir(script)
        #print('Done - returning to '+os.getcwd())

    elif days == 'lastMonth':
        lastmonthdate = pd.to_datetime('today').floor('D') - timedelta(30)
        df = df[df.Date >= lastmonthdate]
        ax = df.plot('Date', 'volume', figsize=(15,10), fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel('Volume', fontsize=20)
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'VolumeLastMonth.png', optimize=True)
        #print('saved '+symbol+'VolumeLastMonth.png to '+os.getcwd())
        os.chdir(symbolFolder)
        #print('returning to '+os.getcwd())
        plt.close()
        ax = df.plot('Date', 'close', figsize=(11,10), fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel('Price', fontsize=20)
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'PriceLastMonth.png', optimize=True)
        #print('saved '+symbol+'PriceLastMonth.png to '+os.getcwd())
        os.chdir(script)
        #print('Done - returning to '+os.getcwd())
        plt.close()

    elif days == 'lastYear':
        lastyeardate = pd.to_datetime('today').floor('D') - timedelta(365)
        df = df[df.Date >= lastyeardate]
        ax = df.plot('Date', 'volume', figsize=(15,10), fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel('Volume', fontsize=20)
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'VolumeLastYear.png', optimize=True)
        #print('saved '+symbol+'VolumeLastYear.png to '+os.getcwd())
        os.chdir(symbolFolder)
        #print('returning to '+os.getcwd())
        plt.close()
        ax = df.plot('Date', 'close', figsize=(11,10), fontsize=15)
        plt.plot(df.Date, df['SMA1'], 'g--', label="Simple Moving Average - 25 Days")
        plt.plot(df.Date, df['SMA2'], 'r--', label="Simple Moving Average - 100 Days")
        ax.set_xlabel('')
        ax.set_ylabel('Price', fontsize=20)
        plt.legend()
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'PriceLastYear.png', optimize=True)
        #print('saved '+symbol+'PriceLastYear.png to '+os.getcwd())
        plt.close()
        os.chdir(script)
        #print('Done - returning to '+os.getcwd())
        
    else:
        ax = df.plot('Date', 'volume', figsize=(15,10), fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel('Volume', fontsize=20)
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'VolumeAllTime.png', optimize=True)
        #print('saved '+symbol+'VolumeAllTime.png to '+os.getcwd())
        plt.close()
        os.chdir(symbolFolder)
        #print('returning to '+os.getcwd())
        ax = df.plot('Date', 'close', figsize=(11,10), fontsize=15)
        plt.plot(df.Date, df['SMA1'], 'g--', label="Simple Moving Average - 25 Days")
        plt.plot(df.Date, df['SMA2'], 'r--', label="Simple Moving Average - 100 Days")
        ax.set_xlabel('')
        ax.set_ylabel('Price', fontsize=20)
        plt.legend()
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True)
        os.chdir(images)
        plt.savefig(symbol+'PriceAllTime.png', optimize=True)
        #print('saved '+symbol+'PriceAllTime.png to '+os.getcwd())
        os.chdir(script)
        #print('Done - returning to '+os.getcwd())
        plt.close()


# In[191]:


def start_visualizations():


    stockSymbol = ["AAPL", "AMZN", "GOOGL","MSFT", "DELL", "IBM", "INTC", "HPQ",
                   "FB", "CSCO", "ORCL", "HPE", "MU", "DXC", "TMO"]

    # initalize relative path directory
    script = os.getcwd()
    images = os.path.join(script, 'visualization', 'WebsitePNGs')
    
    for i in stockSymbol:
        days='lastWeek'
        GetOtherCompanies('{}_twits.csv'.format(i), days, script, images)
        days='lastMonth'
        GetOtherCompanies('{}_twits.csv'.format(i), days, script, images)
        days='lastYear'
        GetOtherCompanies('{}_twits.csv'.format(i), days, script, images)
        days='all'
        GetOtherCompanies('{}_twits.csv'.format(i), days, script, images)
        print('GetOtherCompaniesOver')
        days='lastWeek'
        GetVolume('{}Daily.csv'.format(i), days, script, images)
        days='lastMonth'
        GetVolume('{}Daily.csv'.format(i), days, script, images)
        days='lastYear'
        GetVolume('{}Daily.csv'.format(i), days, script, images)
        days='all'
        GetVolume('{}Daily.csv'.format(i), days, script, images)
        days='lastWeek'
        GetTags('{}_twits.csv'.format(i), days, script, images)
        days='lastMonth'
        GetTags('{}_twits.csv'.format(i), days, script, images)
        days='lastYear'
        GetTags('{}_twits.csv'.format(i), days, script, images)
        days='all'
        GetTags('{}_twits.csv'.format(i), days, script, images)
        print(i)

#print('predictions')
#start_predictions()
#print("compute predictions")
#print(nowEST)
#print(i)
print("visuals")
start_visualizations()
