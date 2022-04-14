# Group Project 2

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



## TECHNICAL- 
Import section with all the required libraries to run the codes:

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from newsapi import NewsApiClient

from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import yfinance as yf
