


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm



def mean_absolute_percentage_error(test, forecast): 
    test, forecast = np.array(test), np.array(forecast)
    return np.mean(np.abs((test - forecast) / test)) * 100


def moving_average(train, test,value):
    #Moving average approach
    y_hat_avg = test.copy()
    windowsize=np.arange(len(train))
    windowsize=windowsize[1:]
    #try to find the lowest error with all possible windowsizes
    error=10000000000000.0
    for window in windowsize:
        y_hat_avg['moving_avg_forecast'] = train[value].rolling(window).mean().iloc[-1]
        mape = mean_absolute_percentage_error(test[value], y_hat_avg.moving_avg_forecast)
        if mape<error:
            error=mape
            optimal_windowsize=window
    return error,optimal_windowsize


def simp_exp_smoothing(train, test,value):
    # Simple Exponential Smoothing
    y_hat_avg = test.copy()
    alphas=np.linspace(0,1,101)
    #try to find the lowest error with all possible alphas with two decimal
    error=100000000000
    for alpha in alphas:
        fit2 = SimpleExpSmoothing(np.asarray(train[value])).fit(smoothing_level=alpha,optimized=False)
        y_hat_avg['SES'] = fit2.forecast(len(test))
        mape = mean_absolute_percentage_error(test[value], y_hat_avg.SES)
        if mape<error:
            error=mape
            optimal_alpha=alpha
    return error,optimal_alpha

def holt_winters(train, test,value,seasons):
    # Holt-Winters Method
    y_hat_avg = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train[value]) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
    mape=mean_absolute_percentage_error(test[value], y_hat_avg.Holt_Winter)
    return mape

def holt(train,test,value):
    sm.tsa.seasonal_decompose(train[value],freq = 3).plot()
    result = sm.tsa.stattools.adfuller(train[value])
    # plt.show()
    y_hat_avg = test.copy()
    alphas=np.linspace(0,1,101)
    #try to find the lowest error with all possible alphas and slopes with two decimal
    error=1000000000000000
    for alpha in alphas:
        slopes=np.linspace(0,1,101)
        for slope in slopes:
            fit1 = Holt(np.asarray(train[value])).fit(smoothing_level =alpha,smoothing_slope =slope)
            y_hat_avg['Holt'] = fit1.forecast(len(test))
            mape=mean_absolute_percentage_error(test[value], y_hat_avg.Holt)
            if mape<error:
                optimal_alpha=alpha
                error=mape
                optimal_slope=slope
    return error,optimal_alpha,optimal_slope



#*************************** WORLD POPULATION **************************

dfWorld=pd.read_csv("World.csv",sep=";")

#In order to get train and test data we use Pareto Principle
size=len(dfWorld)
size_train=round(size*0.8)
train=dfWorld[0:size_train]
test=dfWorld[size_train:]

#moving_averages, simple exponential smoothing, holt-winters,Holt
errors=[0.0,0.0,0.0,0.0]
errors[0]=moving_average(train,test,"POPULATION")[0]
errors[1]=simp_exp_smoothing(train,test,"POPULATION")[0]
errors[2]=holt_winters(train,test,"POPULATION",seasons=10)
errors[3]=holt(train,test,"POPULATION")[0]

optimal_windowsize=moving_average(train,test,"POPULATION")[1]
ses_optimal_alpha=simp_exp_smoothing(train,test,"POPULATION")[1]
holt_optimal_alpha=holt(train,test,"POPULATION")[1]
holt_optimal_slope=holt(train,test,"POPULATION")[2]
 


if errors[0]==min(errors):
    #if moving average method gives the smallest MAPE
    forecast= dfWorld['POPULATION'].rolling(optimal_windowsize).mean().iloc[-1]
    print("Since, Moving Averages method is the best method for World Population data, we applied this method")
    print("Our World Population estimate for the year of 2018 is:",forecast[0])
elif errors[1]==min(errors):
    #if Simple Exponential Smoothing method gives  the smallest MAPE
    fit2 = SimpleExpSmoothing(np.asarray(dfWorld['POPULATION'])).fit(smoothing_level=ses_optimal_alpha,optimized=False)
    forecast=fit2.forecast(1)
    print("Since, Holt method is the best method for World Population data, we applied this method")
    print("Our World Population estimate for the year of 2018 is:",forecast[0])
elif errors[2]==min(errors):
    #if Holt-Winters method gives  the smallest MAPE
    seasons = 10
    fit = ExponentialSmoothing( np.asarray(dfWorld['POPULATION']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    forecast= fit.forecast(1)
    print("Since, Holt-Winters method is the best method for World Population data, we applied this method")
    print("Our World Population estimate for the year of 2018 is:",forecast[0])
else:
    #if Holt method gives  the smallest MAPE
    fit1 = Holt(np.asarray(dfWorld["POPULATION"])).fit(smoothing_level =holt_optimal_alpha,smoothing_slope =holt_optimal_slope)
    forecast=fit1.forecast(1)
    print("Since, Holt method is the best method for World Population data, we applied this method")
    print("Our World Population estimate for the year of 2018 is:",forecast[0])


growth_population=((forecast-dfWorld.POPULATION.iloc[-1])/dfWorld.POPULATION.iloc[-1])*100
print("Growth rate of the world population is",growth_population[0])



#*************************** SOCIAL MEDIAS **************************

dfSocialMedias=pd.read_csv("social_medias.csv",sep=";")

#In order to get train and test data we use Pareto Principle
size=len(dfSocialMedias)
size_train=round(size*0.8)
train=dfSocialMedias[0:size_train]
test=dfSocialMedias[size_train:]

#*************************** TWITTER **************************


#moving_averages, simple exponential smoothing, holt-winters,Holt
t_errors=[0.0,0.0,0.0,0.0]
t_errors[0]=moving_average(train,test,"FACEBOOK")[0]
t_errors[1]=simp_exp_smoothing(train,test,"FACEBOOK")[0]
t_errors[2]=holt_winters(train,test,"FACEBOOK",seasons=10)
t_errors[3]=holt(train,test,"FACEBOOK")[0]

t_optimal_windowsize=moving_average(train,test,"FACEBOOK")[1]
t_ses_optimal_alpha=simp_exp_smoothing(train,test,"FACEBOOK")[1]
t_holt_optimal_alpha=holt(train,test,"FACEBOOK")[1]
t_holt_optimal_slope=holt(train,test,"FACEBOOK")[2]
 


if t_errors[0]==min(t_errors):
    #if moving average method gives the smallest MAPE
    forecast_twitter= dfSocialMedias['TWITTER'].rolling(t_optimal_windowsize).mean().iloc[-1]
    print("Since, Moving Averages method is the best method for Twitter data, we applied this method")
    print("Our Twitter estimate for the last quarter of 2018 is:",forecast_twitter[0])
elif t_errors[1]==min(t_errors):
    #if Simple Exponential Smoothing method gives  the smallest MAPE
    fit2 = SimpleExpSmoothing(np.asarray(dfSocialMedias['TWITTER'])).fit(smoothing_level=t_ses_optimal_alpha,optimized=False)
    forecast_twitter=fit2.forecast(1)
    print("Since, Holt method is the best method for Twitter data, we applied this method")
    print("Our Twitter estimate for the last quarter of 2018 is:",forecast_twitter[0])
elif t_errors[2]==min(t_errors):
    #if Holt-Winters method gives  the smallest MAPE
    seasons = 10
    fit = ExponentialSmoothing( np.asarray(dfSocialMedias['TWITTER']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    forecast_twitter= fit.forecast(1)
    print("Since, Holt-Winters method is the best method for Twitter data, we applied this method")
    print("Our Twitter estimate for the last quarter of 2018 is:",forecast_twitter[0])
else:
    #if Holt method gives  the smallest MAPE
    fit1 = Holt(np.asarray(dfSocialMedias["TWITTER"])).fit(smoothing_level =t_holt_optimal_alpha,smoothing_slope =t_holt_optimal_slope)
    forecast_twitter=fit1.forecast(1)
    print("Since, Holt method is the best method for Twitter data, we applied this method")
    print("Our Twitter estimate for the last quarter of 2018 is:",forecast_twitter[0])


growth_TWITTER=((forecast_twitter-dfSocialMedias.TWITTER.iloc[-4])/dfSocialMedias.TWITTER.iloc[-4])*100
print("Growth rate of TWITTER is",growth_TWITTER[0])



#*************************** FACEBOOK **************************


#moving_averages, simple exponential smoothing, holt-winters,Holt
f_errors=[0.0,0.0,0.0,0.0]
f_errors[0]=moving_average(train,test,"FACEBOOK")[0]
f_errors[1]=simp_exp_smoothing(train,test,"FACEBOOK")[0]
f_errors[2]=holt_winters(train,test,"FACEBOOK",seasons=10)
f_errors[3]=holt(train,test,"FACEBOOK")[0]

f_optimal_windowsize=moving_average(train,test,"FACEBOOK")[1]
f_ses_optimal_alpha=simp_exp_smoothing(train,test,"FACEBOOK")[1]
f_holt_optimal_alpha=holt(train,test,"FACEBOOK")[1]
f_holt_optimal_slope=holt(train,test,"FACEBOOK")[2]
 


if f_errors[0]==min(f_errors):
    #if moving average method gives the smallest MAPE
    forecast_facebook= dfSocialMedias['FACEBOOK'].rolling(f_optimal_windowsize).mean().iloc[-1]
    print("Since, Moving Averages method is the best method for FACEBOOK data, we applied this method")
    print("Our FACEBOOK estimate for the last quarter of 2018 is:",forecast_facebook[0])
elif f_errors[1]==min(f_errors):
    #if Simple Exponential Smoothing method gives  the smallest MAPE
    fit2 = SimpleExpSmoothing(np.asarray(dfSocialMedias['FACEBOOK'])).fit(smoothing_level=f_ses_optimal_alpha,optimized=False)
    forecast_facebook=fit2.forecast(1)
    print("Since, Holt method is the best method for FACEBOOK data, we applied this method")
    print("Our FACEBOOK estimate for the last quarter of 2018 is:",forecast_facebook[0])
elif f_errors[2]==min(f_errors):
    #if Holt-Winters method gives  the smallest MAPE
    seasons = 10
    fit = ExponentialSmoothing( np.asarray(dfSocialMedias['FACEBOOK']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    forecast_facebook= fit.forecast(1)
    print("Since, Holt-Winters method is the best method for FACEBOOK data, we applied this method")
    print("Our FACEBOOK estimate for the last quarter of 2018 is:",forecast_facebook[0])
else:
    #if Holt method gives  the smallest MAPE
    fit1 = Holt(np.asarray(dfSocialMedias["FACEBOOK"])).fit(smoothing_level =f_holt_optimal_alpha,smoothing_slope =f_holt_optimal_slope)
    forecast_facebook=fit1.forecast(1)
    print("Since, Holt method is the best method for FACEBOOK data, we applied this method")
    print("Our FACEBOOK estimate for the last quarter of 2018 is:",forecast_facebook[0])


growth_FACEBOOK=((forecast_facebook-dfSocialMedias.FACEBOOK.iloc[-4])/dfSocialMedias.FACEBOOK.iloc[-4])*100
print("Growth rate of FACEBOOK is",growth_FACEBOOK[0])


#*************************** COMPARING RATIOS **************************

print("Starting to compare the ratios...")

if growth_population>0:
    #Comparing Twitter estimate with World Population estimate
    if growth_TWITTER>0:
        if growth_TWITTER>growth_population+5:
            print("Twitter will have a meaningful growth rate because it is significantly higher than the growth rate of world population")
        elif growth_TWITTER>growth_population+2:
            print("Twitter will have a meaningful growth rate because it is moderately higher than the growth rate of world population")
        else:
            print("Although Twitter will have a growth, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will grow in 2018, Twitter will lose their users")
    #Comparing Facebook estimate with World Population estimate
    if growth_FACEBOOK>0:
        if growth_FACEBOOK>growth_population+5:
            print("Facebook will have a meaningful growth rate because it is significantly higher than the growth rate of world population")
        elif growth_FACEBOOK>growth_population+2:
            print("Facebook will have a meaningful growth rate because it is moderately higher than the growth rate of world population")
        else:
            print("Although Facebook will have a growth, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will grow in 2018, Facebook will lose their users")   
else:
    #Comparing Twitter estimate with World Population estimate
    if growth_TWITTER<0:
        if growth_TWITTER<growth_population-5:
            print("Twitter will have a decreasing growth rate because it is significantly lower than the growth rate of world population")           
        elif growth_TWITTER<growth_population-2:
            print("Twitter will have a decreasing growth rate because it is moderately lower than the growth rate of world population")
        else:
            print("Although Twitter will have a decreasing growth rate, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will decrease in 2018, Twitter will grow")
    #Comparing Facebook estimate with World Population estimate
    if growth_FACEBOOK<0:
        if growth_FACEBOOK<growth_population-5:
            print("Facebook will have a decreasing growth rate because it is significantly lower than the growth rate of world population")           
        elif growth_FACEBOOK<growth_population-2:
            print("Facebook will have a decreasing growth rate because it is moderately lower than the growth rate of world population")
        else:
            print("Although Facebook will have a decreasing growth rate, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will decrease in 2018, Facebook will grow")   