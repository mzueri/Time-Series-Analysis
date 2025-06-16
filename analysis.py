import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

series=pd.read_csv("data/monthly_sales.csv")
#series["Month"]='190' + series['Month'] # data cleaning
series["t"]=list(range(1,len(series)+1))
print(series.head())
series.plot(x="t",y="Sales")
plt.grid(True)
plt.title("Original time series")
plt.show()

# The dataset shows an increasing trend.

# TODO: plot correlogram








# =============================================================================
# Detrend by Differencing ----
# =============================================================================

def get_diff_series(series):
	
    X=series["Sales"].tolist()
    delta_X = list()
    for i in range(len(X)-1):
        delta_X.append(X[i+1] - X[i])
    
    return pd.DataFrame({"t":series["t"].tolist()[:-1],"Sales":delta_X})

diff1_series=get_diff_series(series)
print(diff1_series.head())
diff1_series.plot(x="t",y="Sales")
plt.grid(True)
plt.title("Differenced (1x) time series")
plt.show()

diff2_series=get_diff_series(diff1_series)
print(diff2_series.head())
diff2_series.plot(x="t",y="Sales")
plt.grid(True)
plt.title("Differenced (2x) time series")
plt.show()








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
plt.plot(y)
plt.plot(y_predictions)
plt.title("Predictions to original time series")
plt.show()

y_detrended = detrend(y,y_predictions)
plt.plot(y_detrended)
plt.title("Detrended time series (subtracting predictions)")
plt.show()
# There may be a parabola in the residuals, suggesting that perhaps a polynomial fit may have done a better job.





# TODO
# =============================================================================
# Detrend by smoothing.
# =============================================================================

# ...




# TODO
# Augmented Dickey-Fuller test to test stationarity of a process

# TODO 
# Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)

