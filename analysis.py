import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


series=pd.read_csv("data/monthly_sales.csv")
#series["Month"]='190' + series['Month'] # data cleaning
series["t"]=list(range(1,len(series)+1))
print(series.head())
plt.plot(series["t"],series["Sales"],label="Original Time Series",color="black")
plt.grid(True)
plt.title("Original time series")
plt.legend()
plt.show()

# The dataset shows an increasing trend.




# =============================================================================
# Plot the Autocorrelation Function (ACF)
# =============================================================================

pd.plotting.autocorrelation_plot(series=series["Sales"])
plt.title('Autocorrelation Plot of the Sales')
plt.xlabel('Lag (k)')
plt.ylabel('Autocorrelation Coefficient')
plt.show()

# The horizontal lines in the plot correspond to 95% and 99% confidence bands.
# The dashed line is 99% confidence band.

# If we compute the sample autocorrelations up to lag 40 and find that more than two or three values fall out
# side the bounds of the 95% confidence interval, or that one value falls far outside the bounds, we therefore reject
# the null hypothesis.




# TODO 
# Partial Autocorrelation Function (PACF)




# =============================================================================
# Detrend by Differencing 
# =============================================================================

def get_diff_series(series):
	
    X=series["Sales"].tolist()
    delta_X = list()
    for i in range(len(X)-1):
        delta_X.append(X[i+1] - X[i])
    delta_X.append(np.nan)
    
    return pd.DataFrame({"t":series["t"].tolist(),"Sales":delta_X})

diff1_series=get_diff_series(series)
plt.plot(series["t"], series["Sales"], label="Original Time Series", color="black")
plt.plot(diff1_series["t"], diff1_series["Sales"], label="Differenced (1x)", color='blue')
diff2_series=get_diff_series(diff1_series)
plt.plot(diff2_series["t"], diff2_series["Sales"], label="Differenced (2x)", color='red')
plt.grid(True)
plt.legend()
plt.title("Differenced time series (1x and 2x) vs original time series")
plt.show()

print(diff1_series.head())
print(diff2_series.head())









# =============================================================================
# Detrend by fitting a line. 
# =============================================================================

# A trend is often easily visualized as a line through the observations.
# Linear trends can be summarized by a linear model, and nonlinear trends may be best summarized using a polynomial or other curve-fitting method.

def detrend(y,y_predictions):
    return [y[i]-y_predictions[i] for i in range(len(y))]

# Scikit-learn expects the independent variable (X) to be a 2D array,
# even if it's a single feature. The dependent variable (y) can be 1D.
X = series["t"].tolist()
X = np.reshape(X, (len(X), 1))
y = series["Sales"].tolist()
model = LinearRegression()
model.fit(X, y)

y_predictions = model.predict(X)
plt.plot(series["t"],series["Sales"],label="Original Time Series",color="black")
plt.plot(y_predictions,label="Predictions",color="orange")
y_detrended = detrend(y,y_predictions)
detrended_series=series.copy()
detrended_series["y_detrended"]=y_detrended
plt.plot(detrended_series["t"],detrended_series["y_detrended"],label="Detrended Time Series",color="green")
plt.grid(True)
plt.legend()
plt.title("Detrending (by subtracting predictions from original time series)")
plt.show()
# There may be a parabola in the residuals, suggesting that perhaps a polynomial fit may have done a better job.





# TODO
# =============================================================================
# Detrend by Filtering/Smoothing (moving average)
# =============================================================================

window_size = 3
# Note: .rolling(center=True) calculates the average for the window centered on the current point.
# This means the first (window_size // 2) and last (window_size // 2) points will have NaN for the trend
# because there aren't enough values to center the window.
y_smoothed_series = series['Sales'].rolling(window=window_size, center=True).mean()
y_detrended_series=series["Sales"]-y_smoothed_series
plt.plot(series["t"],series["Sales"],label="Original Time Series",color="black")
plt.plot(series["t"],y_smoothed_series,label="Smoothed Time Series",color="green")
plt.plot(series["t"],y_detrended_series,label="Detrended Time Series",color="blue")
plt.grid(True)
plt.legend()
plt.title("Detrending (by subtracting smoothed time series (moving average))")
plt.show()




# TODO
# Augmented Dickey-Fuller test to test stationarity of a process



