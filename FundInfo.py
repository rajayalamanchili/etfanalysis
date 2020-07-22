"""Module to extract fund info from yahoo

"""


from bs4 import BeautifulSoup
import requests
import re
import os
import itertools
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from pmdarima import auto_arima

class FundInfo:
    
    def get_summary(self, ticker):
        """scrapes table information from risk tab on yahoo finance"""

        outputValuesDictionary = {}

        try:

            # build url
            siteURL = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker

            # read and parse data
            sitedata = requests.get(siteURL).text
            sitedataParsed = BeautifulSoup(sitedata, "html5lib")

            # extract summary statistics
            keys_values = [re.findall(r'.*">(.+?)</.*', str(irow))[0] for irow \
                           in sitedataParsed.find_all("td")]

            values = keys_values[1::2]

            keys = keys_values[::2]
            keys = [re.sub('[^a-zA-Z0-9 ]', '', key).replace(" ", "_").lower() for key in keys]

            outputValuesDictionary = dict(zip(keys, values))
            

        except Exception as e:

            print(ticker + " read exception: " + str(e.__class__) + "occurred")

        return outputValuesDictionary

    def get_etf_risk(self, ticker):
        """scrapes table information from risk tab on yahoo finance"""

        outputValuesDictionary = {}

        try:

            # build url
            siteURL = "https://finance.yahoo.com/quote/" + ticker + "/risk?p=" + ticker

            # read and parse data
            sitedata = requests.get(siteURL).text
            sitedataParsed = BeautifulSoup(sitedata, "html5lib")

            # extract risk statistics
            regex = re.compile('.*Fl\(start\)')
            values = [re.findall(r'.*">(.+?)</span>', str(irow))[0] for irow in sitedataParsed.find_all("span", {"class": regex}) if "Ta" not in str(irow)]

            regex = re.compile('.*Fl\(start\)')
            keys = [re.findall(r'">([^<].+?)<\/span><\/div>', str(irow)) for irow in sitedataParsed.find_all("div", {"class": regex}) if "Ta" not in str(irow)]
            keys = [re.sub('[^a-zA-Z0-9 ]', '', k[0]).replace(" ", "_").lower() for k in keys if len(k)>0] 

            # create keys for three coolumn values
            postFix = ["3Years", "5Years", "10Years"]
            keys = ["_".join(x) for x in itertools.product(keys, postFix)]
            outputValuesDictionary = dict(zip(keys, values))

            return outputValuesDictionary

        except Exception as e:

            print( ticker + " read exception: " + str(e.__class__) + "occurred")

        return outputValuesDictionary

    def get_etf_performance(self, ticker):
        """scrapes table information from performance tab on yahoo finance"""

        outputValuesDictionary = {}

        try:

            # build url
            siteURL = "https://finance.yahoo.com/quote/" + ticker + "/performance?p=" + ticker

            # read and parse data
            sitedata = requests.get(siteURL).text
            sitedataParsed = BeautifulSoup(sitedata, "html5lib")

            # extract trailing returns and annual total return history
            values = [re.findall(r'.*">(.+?)</span>', str(irow))[0] for irow \
                          in sitedataParsed.find_all("span", {"class": "Fl(start)"}) if ("Ta(e)" in str(irow)) and ("div" not in str(irow))]

            keys = [re.findall(r'.*">(.+?)</span>', str(irow))[0] for irow \
                          in sitedataParsed.find_all("span", {"class": "Fl(start)"}) if "Ta(e)" not in str(irow)]
            keys = [re.sub('[^a-zA-Z0-9 ]', '', key).replace(" ", "_").lower() for key in keys]

            outputValuesDictionary = dict(zip(keys, values[::2]))

            # extract performance overview stats
            regex = re.compile('.*Color*')
            keys_values = [re.findall(r'.*">(.+?)</span>', str(irow))[0] for irow in sitedataParsed.find_all("span", {"class": regex})]
            keys_values = keys_values[:6]

            values = keys_values[::2]

            keys = keys_values[1::2]
            keys = [re.sub('[^a-zA-Z0-9 ]', '', key).replace(" ", "_").lower() for key in keys]

            outputValuesDictionary.update(dict(zip(keys, values)))
            

        except Exception as e:

            print( ticker + " read exception: " + str(e.__class__) + "occurred")

        return outputValuesDictionary

    def get_etf_holdings(self, ticker):
        """scrapes table information from holdings tab on yahoo finance"""

        outputValuesDictionary = {}

        try:

            # build url
            siteURL = "https://finance.yahoo.com/quote/" + ticker + "/holdings?p=" + ticker

            # read and parse data
            sitedata = requests.get(siteURL).text
            sitedataParsed = BeautifulSoup(sitedata, "html5lib")

            # extract portfolio composition, equity holdings, bond ratings
            values = [re.findall(r'.*>(.+?)</span>', str(irow))[0] for irow \
                      in sitedataParsed.find_all("span", {"class": "Fl(end)"}) if "Ta" not in str(irow)]

            keys = [re.findall(r'.*">(.+?)</span>', str(irow))[0] for irow \
                    in sitedataParsed.find_all("span", {"class": "Fl(start)"}) if "Ta" not in str(irow)]
            keys = [re.sub('[^a-zA-Z0-9 ]', '', key).replace(" ", "_").lower() for key in keys]


            outputValuesDictionary = dict(zip(keys, values))


            # extract sector weightings
            keys_values = [re.findall(r'.*">(.+?)</span>', str(irow))[0] for irow \
                           in sitedataParsed.find_all("span", {"class": "Fl(start)"}) if ("div" not in str(irow) and "Ta" in str(irow))]

            values = keys_values[1::2]

            keys = keys_values[::2]
            keys = [re.sub('[^a-zA-Z0-9 ]', '', key).replace(" ", "_").lower() for key in keys]

            outputValuesDictionary.update(dict(zip(keys, values)))
            

        except Exception as e:

            print( ticker + " read exception: " + str(e.__class__) + "occurred")

        return outputValuesDictionary
    
    def get_etf_historic_data(self, ticker, startDate, endDate, interval):
        """extracts historical data yahoo finance"""

        siteURL = "https://query1.finance.yahoo.com/v7/finance/download/"+ \
                    ticker + "?period1=" + str(startDate) + "&period2=" + str(endDate) + \
                    "&interval=" + interval + "&events=history"

        # read and parse data
        sitedata = pd.read_csv(siteURL)
        
        # format column names
        sitedata.columns = sitedata.columns.str.replace("\n| ", "_")
        sitedata.columns = [x.lower() for x in sitedata.columns]
        
        # format date column
        #sitedata["date"] = sitedata['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
        
        return sitedata

    def get_etf_info(self, ticker):
        """extracts fund info of given ticker symbol from yahoo and returns data frame"""
        output = {}

        try:
            iteration = 0
            while (iteration < 3):
                
                output = { **{"symbol": [ticker.split(".")[0] if "." in ticker else ticker][0]},
                          **self.get_summary(ticker),
                           **self.get_etf_holdings(ticker),
                           **self.get_etf_performance(ticker),
                           **self.get_etf_risk(ticker)}
                iteration = iteration + 1
                
                if ("previous_close" in output):
                    break

        except Exception as e:

            print(ticker + " read exception: " + str(e.__class__) + "occurred")

        return output


    def get_etf_info_multiple_tickers(self, tickers):
        outputDict = {}

        for ticker in tqdm(tickers):

            if len(outputDict) == 0:
                tempDictionary = self.get_etf_info(ticker)
                outputDict = dict((k, [] + [v]) for (k, v) in tempDictionary.items())

            else: 
                tempDictionary = self.get_etf_info(ticker)
                #outputDict = dict((k, v + ([tempDictionary[k]] if k in tempDictionary else ["N/A"])) for (k, v) in outputDict.items())
                
                for (k, v) in outputDict.items():
                    if k in tempDictionary:
                        outputDict[k] = outputDict[k] + [tempDictionary[k]]
                    else:
                        outputDict[k] = outputDict[k] + ["N/A"]
            
            #print(tempDictionary["symbol"], tempDictionary["previous_close"])

        return outputDict
    
class FundCompositionMetrics(FundInfo):
    
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.inputFileName = rootDir + "tsx-et-fs-mi-g-list-2020-06-22-en.csv"
        self.outputFileName = rootDir + "etf_info.csv"
        self.allFundInfoDF = []
        
        if os.path.exists(self.outputFileName):
            print("Reading ticker data from file: started")
            
            self.allFundInfoDF = pd.read_csv(self.outputFileName, encoding= 'unicode_escape')
            
            noDataTickers = self.allFundInfoDF.loc[self.allFundInfoDF["previous_close"].isnull(), "Ticker"]

            # retry tickers with no data
            if len(noDataTickers) > 0:
                print("Extracting data for tickers with no data")
                for ticker in tqdm(noDataTickers):
                    etfInfoDictionary = (self.get_etf_info(ticker + ".TO"))

                    for key in etfInfoDictionary:
                        self.allFundInfoDF.loc[self.allFundInfoDF["Ticker"] == ticker, key] = etfInfoDictionary[key]
                 
                self.allFundInfoDF.to_csv(self.outputFileName, index=False)
                
            print("Reading ticker data from file: done")
                        
        else:
            print("Reading ticker data from url: started")
            
            self.update_etf_info()
            
            print("Reading ticker data from url: done")
            

    


    def update_etf_info(self):

        # read file with fund tickers
        if os.path.exists(self.inputFileName):
            tickersDF = pd.read_csv(self.inputFileName, skiprows=6, encoding= 'unicode_escape')

            # format names and columns 
            tickersDF = tickersDF.rename(columns={"Root\nTicker":"Ticker",
                                                 " QMV (C$)\n31-May-2020 " : "QuotedMarketValue",
                                                 " O/S Shares\n31-May-2020 " : "OutstandingShares",
                                                 "Date of \nTSX Listing\nYYYYMMDD": "DateOfListing",
                                                 "Place of Incorporation\nC=Canada\nU=USA\nF=Foreign": "PlaceofIncorporation",
                                                 " Volume YTD\n31-May-2020 " : "VolumeYTD",
                                                 " Value (C$) YTD\n31-May-2020 " : "ValueYTD",
                                                 " Number of \nTrades YTD\n31-May-2020 ": "NumberOfTradesYTD",
                                                 " Number of\nMonths of \nTrading Data " : "NumberOfMonths_TradingData"})

            #covert date column to datetime format
            if (not(pd.core.dtypes.common.is_datetime_or_timedelta_dtype(tickersDF["DateOfListing"]))):
                tickersDF["DateOfListing"] = tickersDF['DateOfListing'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

            #remove new lines and spaces in column names
            tickersDF.columns = tickersDF.columns.str.replace("\n| ", "_")

            # read fund info
            selectedTickers = tickersDF["Ticker"][1:]
            #selectedTickers = pd.Series(["TQGM","TULB","VEQT","VCB","VSC","VCN","VEE","VMO","VFV","CHNA","EMV","JAPN","UMI","CAGG"])
            fundInfoDictionary = self.get_etf_info_multiple_tickers(selectedTickers + ".TO")
            
            dropKeys = ["sectors", "average", "monthly_total_returns", "year"]
            for key in dropKeys:
                if key in fundInfoDictionary:
                    fundInfoDictionary.pop(key)
                    
            fundInfoDF = pd.DataFrame(fundInfoDictionary)
            tickersDF = tickersDF[tickersDF["Ticker"].isin(selectedTickers)]

            # save file to csv
            self.allFundInfoDF = tickersDF.merge(fundInfoDF, left_on='Ticker', right_on='symbol')
            self.allFundInfoDF.to_csv(self.outputFileName, index=False)
            print("Info file saved ....")

        else:
            print("list of tickers file is missing")
            print(self.inputFileName)
            
            
class FundPricePrediction(FundInfo):
    
    def __init__(self, ticker, startDate, endDate):
        
        self.ticker = ticker
        
        if(len(startDate) == 0):
            start_date = round((datetime.today() - timedelta(days=180)).timestamp())
        else:
            start_date = str(round(datetime.strptime(startDate, '%Y-%m-%d').timestamp()))
        
        if(len(endDate) == 0):
            end_date = round(datetime.today().timestamp())
        else:
            end_date = str(round(datetime.strptime(endDate, '%Y-%m-%d').timestamp()))
        
        self.valuesDF = self.get_etf_historic_data(ticker + ".TO", 
                                                           startDate = start_date, 
                                                           endDate = end_date, 
                                                           interval = "1d")
        
        self.trainData_x = []
        self.trainData_y = []
        self.validationData_x = []
        self.validationData_y = []
        self.predictions = []
        
        self.prepareData(trainSize=0.7)
        
    def prepareData(self, trainSize):
        
        priceDF = self.valuesDF[["date", "close"]]
        
        priceDF["date"] = pd.to_datetime(priceDF["date"],format="%Y-%m-%d")
        priceDF.index = priceDF["date"]

        priceDF = priceDF.sort_index(ascending=True, axis=0)

        priceDF["year"]=priceDF["date"].dt.year 
        priceDF["month"]=priceDF["date"].dt.month 
        priceDF["week"]=priceDF["date"].dt.week
        priceDF["day"]=priceDF["date"].dt.day

        priceDF["dayofweek"]=priceDF["date"].dt.dayofweek
        priceDF["dayofyear"]=priceDF["date"].dt.dayofyear

        priceDF["is_month_start"]=priceDF["date"].dt.is_month_start
        priceDF["is_month_end"]=priceDF["date"].dt.is_month_end
        priceDF["is_quarter_start"]=priceDF["date"].dt.is_quarter_start
        priceDF["is_quarter_end"]=priceDF["date"].dt.is_quarter_end
        priceDF["is_year_start"]=priceDF["date"].dt.is_year_start
        priceDF["is_year_end"]=priceDF["date"].dt.is_year_end
        
        
        trainSize = round(len(priceDF)*trainSize)

        self.trainData_x = priceDF[:trainSize].drop(["date", "close"], axis=1)
        self.trainData_y = priceDF[:trainSize]["close"]

        self.validationData_x = priceDF[trainSize:].drop(["date", "close"], axis=1)
        self.validationData_y = priceDF[trainSize:]["close"]

        print("Training samples: %d \nValidation samples: %d"%(len(self.trainData_x), len(self.validationData_x)))
        
    def predictMovingAverage(self, windowSize):
        predictions = []

        for i in range(len(self.validationData_y)):
            predictions.append((self.trainData_y[len(self.trainData_y)-windowSize+i:].sum() + sum(predictions))/windowSize)
            
        self.predictions = predictions
        
        self.plotData()
        
    def predictRegression(self, option="linear"):
        predictions = []
        
        if (option == "linear"):
            model = LinearRegression()
        elif (option == "ridge"):
            model = Ridge(alpha = 0.01)
        elif (option == "lasso"):
            model = Lasso(alpha = 0.01, tol=0.01)
        elif (option == "elasticnet"):
            model = ElasticNet(alpha = 0.01, tol=0.01)
        elif (option == "arima"):
            model = auto_arima(self.trainData_y, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0,
                               seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
        
        if(option == "arima"):
            model.fit(self.trainData_y)
            self.predictions = model.predict(n_periods=len(self.validationData_x))
        else :
            model.fit(self.trainData_x,self.trainData_y)
            self.predictions = model.predict(self.validationData_x)
        
        self.plotData()
    
    def plotData(self):
        
        fig, ax = plt.subplots(figsize = (12, 10))
        plt.plot(self.trainData_y, label="train")
        plt.plot(self.validationData_y, label="validation")
        plt.plot(self.validationData_y.index, self.predictions, label="predictions")
        plt.legend()
        plt.title(self.ticker + "\nMean sqaured error: %.4f"%(mean_squared_error(self.validationData_y, self.predictions)))
        plt.show()