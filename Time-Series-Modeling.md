This project explores Zillow data (https://www.zillow.com/research/data/) with the aim of answering a relatively simple question: what are the top 5 zipcodes to invest in? This looks at a housing dataset from 1996-2018, with location and price info. To make this more manageable, I focused on only the Brooklyn area looking at 2004-2018. 

## 1. Data Acquisition & Shaping


```python

# Import Libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import itertools
import statsmodels.api as sm
from matplotlib.pylab import rcParams
import statsmodels.api as sm
import geopandas as gpd
import descartes
import ffmpy
import imageio
from pandas.tools.plotting import andrews_curves
plt.style.use('ggplot')
```

The data at initial glance is messy; it is situationed so time is listed as columns, and there also seems to be a lot more information than necessary. For the sake of scope (and home-field advantage), I scaled the dataset down to consist of only Brooklyn zip-codes (28 total):

```python
nystate = zillow.loc[zillow['State'].isin(['NY'])] 
nystate = nystate.loc[nystate['CountyName'].isin(['Kings'])]
```
Next I needed to reshape the dataframe from wide to long format, which I approached by using the pandas `melt` function, resulting in a time variable that I set as the index and converted to datetime. 

```python
columns = ["RegionID", "RegionName", "City", "State", "Metro", "CountyName", "SizeRank"]
df = pd.melt(nystate, columns, var_name = 'time')

df.time = pd.to_datetime(df.time)

df_bk = df.dropna(subset = ['value'])

df_bk.set_index('time', inplace = True)
```

Since I was to be only focusing on how time affected house price, I dropped excess columns, resulting in a dataframe that looked like so:

| time                |   RegionName |   value |
|---------------------|--------------|---------|
| 1996-04-01 00:00:00 |        11226 |  162000 |
| 1996-04-01 00:00:00 |        11235 |  190500 |
| 1996-04-01 00:00:00 |        11230 |  230100 |
| 1996-04-01 00:00:00 |        11209 |  255700 |
| 1996-04-01 00:00:00 |        11212 |  125000 |

## 2. Exploration and Visualizing Data

To start, I looked at every zipcode over time to see if there were any glaring visual trends:

![All_Zips](https://luicyfruit.github.io/img/all_zips.png)

Right away we can see that we're missing data for some zipcodes pre-2004, which I then subsequently dropped. Another observation is how much more spread the data gets as you move forward in time - knowing Brooklyn this is definitely on trend with gentrification and groups moving into Brooklyn as the rent prices in Manhattan continue to rise.

I wanted to play with some GIS data to see how data looks spatially - this required getting GIS zipcode data from public resouces, filtering it so it only contained brooklyn, and then merging it with the zillow data:

```python
fp = 'ZIPS/ZIP_CODE_040114.shp'
map_df = gpd.read_file(fp)
map_df= map_df.loc[map_df['COUNTY'].isin(['Kings'])]
map_df['ZIPCODE'] = map_df['ZIPCODE'].astype('int64')
```

After checking the length of the map_df unique zipcodes, which resulted in 40 zipcodes, right away I could see there would be some missing data in the visualization as the zillow dataset only had 29. When plotting the initial GIS data, you get a map of brooklyn zipcodes:

![brooklyn](https://luicyfruit.github.io/img/brooklyn.png)

In order to create maps of each year, I will have to left join subsets of my dataset(by year) with the GIS map dataframe. I created a plotting function so that plotting would be faster. My approach afterwards is to create a map for every year in the dataset (median value is grouped by year average). Afterwards, I can use those maps as "frames" to create an animation that shows the change in median income over time, using the `imageio` package. 

```python
def map_plotting(df, variable, year):
    variable = variable
    # set the range for the choropleth
    vmin, vmax = 120, 220
    # create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axis('off')
    ax.set_title(year, fontdict={'fontsize': '25', 'fontweight' : '3'})
    df.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor= '0.8')
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    cbar = fig.colorbar(sm)
    variable = 'Median'
    fig.savefig(year+'.png', dpi=300)
```

```python
years = ['2004', '2005', '2006', '2007', '2008', '2009', '2010', 
        '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']

# For loop to select data by each year and save a map 
for year in years:
    df = ny_2004[year].groupby('ZIPCODE').resample('AS').mean()
    #merging the two dataframes
    m = map_df.merge(df, how='left', on="ZIPCODE")
    map_plotting(m, 'Median', year)
```

```python
# Turn frame files into a gif
images = []
filenames = ['2004.png', '2005.png', '2006.png', '2007.png', '2008.png', '2009.png', '2010.png', 
        '2011.png', '2012.png', '2013.png', '2014.png', '2015.png', '2016.png', '2017.png', '2018.png']

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images, duration = 1)
```
The resulting gif:

![Gif](https://luicyfruit.github.io/img/movie.gif)

It's interesting to see how the disparity changes as time moves forward. Clearly there are a few zipcodes that are consistently higher than the others, but as time moves forward they continue to stay/get bluer, while the others remain a light blue. The lightest blue zipcodes that never change are the 11 zipcodes that we have no data for. 

## 3. Arima Modeling

After exploring the data, It's clear that some zipcodes have drastically changed more relative to other zipcodes. In order to test which the best "investments" will be, I will use ARIMA models on each of the zipcode subsets. I wanted to use functions to manipulate the data and to keep things cleaner. I created a sampling function, a pdq parameter function, an arima modeling function, a mean squared error function, a predictor function, and a plotting function. This way I would be able to call these functions on multiple data subsets (different zipcodes) faster and cleaner

### Function Creation
**Sampling Function**: This only returns the median $ values and keeps the time index. We resample at the Month to avoid errors, even though the data is already monthly

```python
def sampling(df, z):
    df2 = df.loc[df['ZIPCODE'] == z]
    df2 = df2[['Median']]
    df2 = df2['Median'].resample('MS').mean()
    return df2
```

**PDQ Parameter Function**: This uses the AIC in order to determine the best p, d, and q values. It takes in a dataframe of time and values. It returns p, which has the attributes, pdq, pdqd, and aic. After some exploration, I used the second minimum aic value of all aic values tested for optimum results. 

```python
def pdqz(df):
    p = d = q = range(0, 2)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))
    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    #Parse through combinations
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = sm.tsa.statespace.SARIMAX(df,
                                            order=comb,
                                            seasonal_order=combs,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

                output = mod.fit()
                ans.append([comb, combs, output.aic])
            except:
                continue
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    sorts = ans_df.sort_values(by=['aic'])
    # attributes are pdq, pdqs, and aic
    # after playing with the data a few times I found the second to minimum AIC value to have the best p results
    return sorts.iloc[1]
```

**ARIMA Modelling Function**:This function takes in the dataframe, and the results of the AIC pdqs, and applies them to an arima model. It returns the output of the model. 

```python
def arimamodel(df, p):
    ARIMA_MODEL = sm.tsa.statespace.SARIMAX(df,
                                order= p['pdq'],
                                seasonal_order=p['pdqs'],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    output = ARIMA_MODEL.fit()
    return output
```
**Mean Squared Error Function**: This function takes in the dataframe, the results of teh ARIMA, and the date at which you want the ARIMA to start. It returns the mean squared error of the ARIMA predictions, vs the actual data. 

```python
def MSEr(df, output, date):
    pred = output.get_prediction(start=pd.to_datetime(date), dynamic=False)
    pred_conf = pred.conf_int()
    forecasted = pred.predicted_mean
    truth = df[date:]
    mse = ((forecasted - truth) ** 2).mean()
    return mse
```
**Futuremost Value**: This function takes in the dataframe, the results of the ARIMA, and the number of steps into the future. It returns the mean value at the most recent step, the upper bound, and the lower bound of the confidence.

```python
def future_value(df, output, steps):
    # Get forecast X steps ahead in future
    prediction = output.get_forecast(steps= steps)
    mean = prediction.predicted_mean[-1]
    # Get confidence intervals of forecasts
    pred_conf = prediction.conf_int()
    lower = pred_conf['lower Median'][-1]
    upper = pred_conf['upper Median'][-1]
    return mean, lower, upper
```
**Plot Forecasts**: This function takes in the dataframe, the results of the arima, the amount of time in the future, the start date of your predictions, the x-axis label, and the y-axis label. It returns a graph with the observed and the forecasted results

```python
def plotting_forecasts(df, output, steps, date, x, y):
    prediction = output.get_forecast(steps = steps)
    pred_conf = prediction.conf_int()

    #Plot observed values
    ax = df.plot(label='observed', figsize=(20, 15))
    prediction.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_conf.index,
                    pred_conf.iloc[:, 0],
                    pred_conf.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.legend()
    plt.show()
```

A quick test of these functions (using my home zip code) results in a plot of time series prediction, and diagnostics
```python
test = sampling(ny_2004, 11211)
p = pdqz(test)
output = arimamodel(test, p)
mse = MSEr(test, output, '2010-01-01' )
futures = future_value(test, output, 24)
plotting_forecasts(test, output, 24, '2010-01-01', 'Date', 'Median House Value')
output.summary()
```

![WB](https://luicyfruit.github.io/img/wb.png)

### ARIMA Modelling

Now that all the fuctions are defined and working, my next step is to create a dataset with the ARIMA predictions to see what the best zipcodes in Brooklyn are to invest in. I will run this with 2 years from the end of the dataset (so 2020 predictions), and will record the zipcode, mean squared error, value at 2 years, value 5 years previous, value 10 years previous, the lower confidence interval, and the upper confidence interval. I then calculated the 10 year expected growth, 5 year expected growth, and 5 and 10 year risky and conservative growth. 

```python
## Create empty dataframe to store results in
results = pd.DataFrame({'zipcode': [], 'mse': [], 'mean_value': [], '5YA': [], '10YA': [],'lower': [], 'upper':[] })

## For Loop to run an arima model on each zipcode, grouped by zipcode. 
for i in zip_reals:
    # Create dataframe for that zipcode
    df = sampling(ny_2004, i)
    # Get ARIMA parameters
    p = pdqz(df)
    # Run ARIMA Model
    output = arimamodel(df, p)
    # Get MSE   
    mse = MSEr(df, output, '2010-01-01')
    # 5 & 10 Yr Values
    five = df['2015-01-01']
    ten = df['2010-01-01']
    # Get 12 steps ahead value and confidence interval
    futures = future_value(df, output, 24)
    mean = futures[0]
    lower = futures[1]
    upper = futures[2]
    # Store values in new dataframe
    results = results.append({'zipcode': i, 'mse': mse, 'mean_value': mean, '5YA': five, '10YA': ten,'lower': lower, 'upper':upper }, ignore_index=True)
 ```
 
 ```python
results['growth_10'] = results['mean_value'] - results['10YA']
results['conservative10'] = results['lower'] - results['10YA']
results['risk10'] = results['upper'] - results['10YA']
results['growth_5'] = results['mean_value'] - results['5YA']
results['conservative5'] = results['lower'] - results['5YA']
results['risk5'] = results['upper'] - results['5YA']
```

## Interpreting Results

My method for choosing the top five zipcodes to invest in was to look at all of the growth parameters for both five and ten year growth. There was immediately a distinct pattern: for both 5 and 10 year growth, the best zipcodes to invest in remained the same, across conservative and risky growth. Overall, consistently 11217, 11238, 11211, 11216, and 11222 as the top 5 zipcodes to invest in. These correspond with Boreum Hill, Prospect Park, Williamsburg (my home!) Bed-Stuy, and Greenpoint. This is consistent with my local knowledge of the areas, as they’re known to be places with an increasing younger demographic looking for places with more square footage for the $. Seeing these trends in these places would be great if you were a home-owner or land-lord (or lived in a rent-controlled apartment), but isn’t so great for regular renters like me! There are a few other players, (11223, 11215, and 11230) which could also be considered, however this would be more dependent on investment preferences. 



