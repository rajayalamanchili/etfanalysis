"""Module to extract fund info from yahoo

"""


from bs4 import BeautifulSoup
import requests
import re
import os
import itertools
from tqdm import tqdm
import pandas as pd

class FundInfo:
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.inputFileName = rootDir + "tsx-et-fs-mi-g-list-2020-06-22-en.csv"
        self.outputFileName = rootDir + "etf_info.csv"
        self.allFundInfoDF = []
        
        if os.path.exists(self.outputFileName):
            self.allFundInfoDF = pd.read_csv(self.outputFileName, encoding= 'unicode_escape')
        else:
            self.update_etf_info()
            

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

    def get_etf_info(self, ticker):
        """extracts fund info of given ticker symbol from yahoo and returns data frame"""
        output = {}

        try:
            
            output = { **{"symbol": [ticker.split(".")[0] if "." in ticker else ticker][0]},
                      **self.get_summary(ticker),
                       **self.get_etf_holdings(ticker),
                       **self.get_etf_performance(ticker),
                       **self.get_etf_risk(ticker)}

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

        return outputDict


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
            selectedTickers = tickersDF["Ticker"][1:100]
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