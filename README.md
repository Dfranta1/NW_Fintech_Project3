# Group Project 3

## INTRODUCTION:
The focus of this project is to analyze stocks using 30 day data, applying latest news for sentiment analysis to see whether it is positive or negative indicators.
The idea is to check out news articles with the help of news API which were published within specified timeframe, 
then with the help of Natural Language Processing(NLP) techniques such as sentiment analysis we can find out (positive, negative or neutral) indicators. 
We are using Yfinance to pull the data. We will be using S&P 500 for stock analysis. Tickers were separated from Dataframe and adjusted close price used for visualization. 


## DATA PREPARATION:
With API News pull we were able to get only one month of historical data for free. 
Pulled stock data using Yfinance for 90 days(from january 19, 2022 through april 12, 2022). 
After that we created a "returns" column by using the pct_change function on the “Adj Close” column with a -1 shift =”returns_df”.
Then we merged two dataframes df_stock_data and returns_df using pd.concat function 
Prices were converted to percentage in order to get a more precise description as how the data has changed over period of time. 

RANDOM FOREST- Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.

benefits of random forest:
A random forest produces good predictions that can be understood easily.
It can handle large datasets efficiently. and gives high level of accuracy on predicting outcomes.


LOGISTIC REGRESSION- is a supervised learning algorithm used in machine learning to predict the probability of a binary outcome. Logical regression is used in predictive modeling to analyze large datasets in which one or more independent variables can determine an outcome.
the outcome is limited to one of the two possible outcomes: its either yes/no, true/false, 1/0.

We used the VADER Sentiment analysis program to analyze our News API articles. VADER stands for: “Valence Aware Dictionary and Sentiment Reasoner”
VADER divides the sentiment into three categories. Positive; Neutral; and Negative, from which it generates four scores: Positive (pos); Negative (neg); Neutral (neu) and a Compound Score.


## TECHNICAL- 
Import section with all the required libraries to run the codes:

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from newsapi import NewsApiClient

import os

from dotenv import load_dotenv, find_dotenv

from datetime import date, timedelta, datetime

import hvplot.pandas

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import yfinance as yf

import talib

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from collections import Counter

# Definitions of Technical Indicators Used from TA LIB

**Simple Moving Average (SMA)** is one of the core indicators in technical analysis, SMA is the easiest moving average to construct. It's the average price over the specified period.

**Exponential Moving Average(EMA)** is similar to simple moving average measures a trend direction over a period of time and applies a weight to data that is more current.

**Momentum (MOM):** measures the velocity of price change over a given period of time

**Average Directional Movement Index (ADX)** can be used to help measure the overall strength of a trends. It is a component of the Directional Movement System developed by Welle Wilder.

**Normalized Average True Range:** attempts to normalize the average true range values across instruments by using the formula below.

**Linear Regression (LINEARREG)** plots the ending value of a linear regression line for a specified number of bars, showing statistically, where the price is expected to be for a period. Used in conjunction with Simple Moving Average

**HT_TRENDMODE Hilbert Transform Trend vs Cycle Mode:** this trend vs. cycle is a binary indicator in the sense that it will show either a value of 1 or 0. The interpretation is very simple – when the indicator switches from 0 to 1 it means that a new trend started, it entered a trending phase.

**RSI - Relative Strength Index:** measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.

**TYPPRICE typical price, builder of Money Flow Index(MFI)** provides a simple, single-line plot of the day’s average price. Some investors use the Typical Price rather than the closing price when creating moving-average penetration systems. The Typical Price is a building block of the Money Flow Index and is calculated by adding the high, low, and closing prices together, and then dividing by three.

**MFI - Money Flow Index:** a technical oscillator that uses price and volume data for identifying overbought or oversold signals in an asset. It can also be used to spot divergences which warn of a trend change in price. The oscillator moves between 0 and 100.

**ADOSC-Chaikin A/D Oscillator, Chaikin Oscillator:** The Chaikin Oscillator examines both the strength of price moves and underlying buying and selling pressure.
It provides a reading of the demand for a security, and possible turning points in the price.

**HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period:** HTPeriod (or MESA Instantaneous Period) returns the period of the Dominant Cycle of the analytic signal as generated by the Hilbert Transform. The Dominant Cycle can be thought of as being the "most likely" period (in the range of 10 to 40) of a sine function of the Price Series.

**HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase:** The Hilbert Transform is a technique used to generate inphase and quadrature components of a de-trended real-valued "analytic-like" signal (such as a Price Series) in order to analyze variations of the instantaneous phase and amplitude.. HTDCPhase returns the Hilbert Transform Phase of the Dominant Cycle. The Dominant Cycle Phase lies in the range of 0 to 360 degrees.


Sources

[Investopedia](https://www.investopedia.com/)

[Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/overview)

[TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/doc_index.html)


