import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf 
from scipy.optimize import minimize


## step 1: data retrival 


class ARCH: 
    def __init__(self, p = 1):
        self.p = p
        self.alpha0 = None
        self.alphas = None
        self.conditional_variance = None
    
    def _conditional_variance(self, params, returns):
        alpha0 = params[0]
        alphas = params[1:]
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[:self.p] = np.var(returns)

    # Add epsilon to prevent sigma2 from being zero or negative
        epsilon = 1e-8
        for t in range(self.p, n):
            sigma2[t] = alpha0 + np.sum(alphas * returns[t-self.p:t]**2)
            sigma2[t] = max(sigma2[t], epsilon)  # Ensure positivity

        return sigma2


    def _log_likelihood(self, params, returns):
        """
        Compute the negative log-likelihood function.
        """
        sigma2 = self._conditional_variance(params, returns)
        log_likelihood = -np.sum(-np.log(sigma2) - (returns**2 / sigma2))
        return log_likelihood

    def fit(self, returns):
        # Use realistic initial guesses
        initial_alpha0 = 0.1 * np.var(returns)  # Initial constant term
        initial_alphas = [0.2] * self.p  # Initial ARCH coefficients
        initial_params = np.array([initial_alpha0] + initial_alphas)

        # Optimization routine
        result = minimize(self._log_likelihood, initial_params, args=(returns,), method='L-BFGS-B')
        
        if result.success:
            self.alpha0 = result.x[0]
            self.alphas = result.x[1:]
            self.conditional_variance = self._conditional_variance(result.x, returns)
            print("Fitted Parameters:", result.x)
        else:
            raise ValueError("Optimization failed:", result.message)


    def forecast(self, steps=5):
        """
        Forecast future volatility.
        
        Parameters:
        steps (int): Number of steps to forecast.
        """
        forecasts = []
        last_sigma2 = self.conditional_variance[-1]

        for _ in range(steps):
            forecast_sigma2 = self.alpha0 + np.sum(self.alphas * last_sigma2)
            forecasts.append(np.sqrt(forecast_sigma2))
            last_sigma2 = forecast_sigma2
        
        return forecasts  
    
class ARIMA:
    def __innit__(self, p =1, d = 1, q = 1):
        self.p = p
        self.alpha0 = None
        self.alphas = None
        self.conditional_variance = None
        
    def autoregressive(series, lags):
        
        n = len(series)
        X = []
        y = []
        
        for t in range(lags, n):
            X.append(series[t - lags:t].values)
            y.append(series[t])
        
        X = np.array(X)
        y = np.array(y)
        
        phi = np.linalg.lstsq(X,y,rcond=None)[0]
        return phi
    
    def moving_average(errors, q):
        n = len(errors)
        X = []
        y = []
        
        for t in range(q,n):
            X.append(errors[t-q:t])
            y.append(errors[t])
            
        X = np.array(X)
        y = np.array(y)
        
        theta = np.linalg.lstsq(X,y,rcond=None)[0]
        return theta
    
    
    def arima_forecast(series, ar_coeffs, ma_coeffs, lags, q, steps):
        """
        Forecast future values using ARIMA components.
        """
        predictions = list(series[-lags:])  # Start with the last 'lags' values
        errors = [0] * q  # Initialize past errors

        for _ in range(steps):
            # AR component: predict using past values
            ar_part = np.dot(ar_coeffs, predictions[-lags:])

            # MA component: use past errors
            ma_part = np.dot(ma_coeffs, errors[-q:])

            # Combine AR and MA parts
            prediction = ar_part + ma_part
            predictions.append(prediction)

            # Simulate an error (assumed to be 0 for forecast)
            error = 0
            errors.append(error)

        return predictions[-steps:]

class SARIMA:
    def __init__(self, p, d, q, P=0, D=0, Q=0, m=1):
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.params_ = None
        
    def _difference(self, y, d=0, m=1):
        # Simple differencing function: 
        # non-seasonal differencing if m=1, else seasonal differencing
        # For seasonal differencing: Y_t - Y_(t-m)
        # For non-seasonal: Y_t - Y_(t-1)
        if d == 0:
            return y
        diffed = y
        for _ in range(d):
            diffed = diffed[m:] - diffed[:-m]
        return diffed

    def _prepare_data(self, y):
        # Apply non-seasonal differencing
        yd = self._difference(y, self.d, 1)
        # Apply seasonal differencing
        yd = self._difference(yd, self.D, self.m)
        return yd

    def _loglik(self, params, y):
        # Params structure (rough guess):
        # AR non-seasonal: p terms
        # MA non-seasonal: q terms
        # AR seasonal: P terms
        # MA seasonal: Q terms
        # plus variance (sigma^2)
        
        # A simple parameter parsing scheme (not robust):
        np_ar = self.p
        np_ma = self.q
        np_arS = self.P
        np_maS = self.Q
        
        ar_params = params[0:np_ar]
        ma_params = params[np_ar:(np_ar+np_ma)]
        arS_params = params[(np_ar+np_ma):(np_ar+np_ma+np_arS)]
        maS_params = params[(np_ar+np_ma+np_arS):(np_ar+np_ma+np_arS+np_maS)]
        sigma2 = params[-1]

        if sigma2 <= 0:
            return 1e9  # penalize invalid sigma2

        n = len(y)
        # We'll simulate errors using a basic linear process:
        # y_t = AR terms + seasonal AR terms + error + ...
        
        # Initialize
        maxlag = max(self.p, self.q, self.P*self.m, self.Q*self.m)
        if maxlag == 0:
            maxlag = 1
        residuals = np.zeros(n)
        
        for t in range(maxlag, n):
            # Compute AR part
            ar_part = 0
            for i in range(self.p):
                ar_part += ar_params[i] * y[t-1-i] if t-1-i>=0 else 0
            # Seasonal AR part
            for i in range(self.P):
                ar_part += arS_params[i] * y[t-(i+1)*self.m] if t-(i+1)*self.m>=0 else 0
            
            # Compute MA part: approximate by residuals
            ma_part = 0
            for j in range(self.q):
                ma_part += ma_params[j] * residuals[t-1-j] if t-1-j>=0 else 0
            # Seasonal MA part
            for j in range(self.Q):
                ma_part += maS_params[j] * residuals[t-(j+1)*self.m] if t-(j+1)*self.m>=0 else 0
            
            # residual
            residuals[t] = y[t] - (ar_part + ma_part)
        
        # Gaussian log-likelihood
        ll = -0.5*n*np.log(2*np.pi) - 0.5*n*np.log(sigma2) - 0.5*np.sum((residuals**2)/sigma2)
        return -ll  # minimize negative log-likelihood

    def fit(self, y):
        yd = self._prepare_data(y)
        
        # initial guess for parameters
        # ARMA(p,q) + seasonal => number of parameters plus sigma2
        initial_params = np.concatenate([0.1*np.ones(self.p),
                                         0.1*np.ones(self.q),
                                         0.1*np.ones(self.P),
                                         0.1*np.ones(self.Q),
                                         [np.var(yd)]])
        
        res = minimize(self._loglik, initial_params, args=(yd,), method='BFGS')
        self.params_ = res.x
        return self
    
    def predict(self, y, steps=1):
        # A very naive prediction approach using the fitted parameters.
        yd = self._prepare_data(y)
        np_ar = self.p
        np_ma = self.q
        np_arS = self.P
        np_maS = self.Q
        
        ar_params = self.params_[0:np_ar]
        ma_params = self.params_[np_ar:(np_ar+np_ma)]
        arS_params = self.params_[(np_ar+np_ma):(np_ar+np_ma+np_arS)]
        maS_params = self.params_[(np_ar+np_ma+np_arS):(np_ar+np_ma+np_arS+np_maS)]
        
        # Residual calculation as in fit
        n = len(yd)
        maxlag = max(self.p, self.q, self.P*self.m, self.Q*self.m)
        if maxlag == 0:
            maxlag = 1
        residuals = np.zeros(n)
        
        for t in range(maxlag, n):
            ar_part = sum(ar_params[i]*yd[t-1-i] for i in range(np_ar) if t-1-i>=0)
            ar_part += sum(arS_params[i]*yd[t-(i+1)*self.m] for i in range(np_arS) if t-(i+1)*self.m>=0)
            ma_part = sum(ma_params[j]*residuals[t-1-j] for j in range(np_ma) if t-1-j>=0)
            ma_part += sum(maS_params[j]*residuals[t-(j+1)*self.m] for j in range(np_maS) if t-(j+1)*self.m>=0)
            residuals[t] = yd[t] - (ar_part + ma_part)
        
        forecasts = []
        future_y = list(yd)
        future_res = list(residuals)
        
        for h in range(steps):
            t = len(future_y)
            ar_part = sum(ar_params[i]*future_y[t-1-i] for i in range(np_ar) if t-1-i>=0)
            ar_part += sum(arS_params[i]*future_y[t-(i+1)*self.m] for i in range(np_arS) if t-(i+1)*self.m>=0)
            # For forecast, we assume future residuals are zero (best guess)
            pred = ar_part
            forecasts.append(pred)
            future_y.append(pred)
            future_res.append(0.0)
        
        return np.array(forecasts)

class ARCH:
    def __init__(self, q=1):
        self.q = q
        self.params_ = None  # [omega, alpha_1, ..., alpha_q]

    def _loglik(self, params, y):
        omega = params[0]
        alpha = params[1:]
        
        if omega <= 0 or np.any(alpha < 0):
            return 1e9  # impose positivity

        n = len(y)
        # Estimate residuals as just y itself (assuming mean zero)
        eps = y
        sigma2 = np.zeros(n)
        # initialization of sigma2
        sigma2[:self.q] = np.var(y)
        
        for t in range(self.q, n):
            sigma2[t] = omega + np.sum(alpha * eps[t-self.q:t]**2)

        # Gaussian log-likelihood
        ll = -0.5*np.sum(np.log(2*np.pi) + np.log(sigma2) + (eps**2)/sigma2)
        return -ll

    def fit(self, y):
        initial_params = np.ones(self.q+1)*0.1
        initial_params[0] = np.var(y)*0.1  # omega initial
        res = minimize(self._loglik, initial_params, args=(y,), method='BFGS')
        self.params_ = res.x
        return self

    def predict(self, y, steps=1):
        # Predict future variance (not necessarily future mean)
        omega = self.params_[0]
        alpha = self.params_[1:]
        
        eps = y
        n = len(eps)
        sigma2 = np.zeros(n)
        sigma2[:self.q] = np.var(eps)
        for t in range(self.q, n):
            sigma2[t] = omega + np.sum(alpha * eps[t-self.q:t]**2)
        
        forecasts = []
        # For forecasting, we assume future eps=0
        current_sigma = sigma2[-1]
        for h in range(steps):
            # ARCH(q) with no future shocks assumed:
            # sigma2_{t+h} = omega + sum(alpha_i * 0)
            current_sigma = omega
            forecasts.append(current_sigma)
        return np.array(forecasts)

class GARCH:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.params_ = None  # [omega, alpha_1,...,alpha_q, beta_1,...,beta_p]

    def _loglik(self, params, y):
        omega = params[0]
        alpha = params[1:1+self.q]
        beta = params[1+self.q:]
        
        if omega <= 0 or np.any(alpha < 0) or np.any(beta < 0):
            return 1e9

        n = len(y)
        eps = y
        sigma2 = np.zeros(n)
        sigma2[:max(self.p,self.q)] = np.var(eps)
        
        for t in range(max(self.p,self.q), n):
            sigma2[t] = omega
            for i in range(self.q):
                sigma2[t] += alpha[i]*eps[t-1-i]**2
            for j in range(self.p):
                sigma2[t] += beta[j]*sigma2[t-1-j]

        ll = -0.5*np.sum(np.log(2*np.pi) + np.log(sigma2) + (eps**2)/sigma2)
        return -ll

    def fit(self, y):
        init_params = np.array([np.var(y)*0.1] + [0.05]*self.q + [0.9]*self.p)
        res = minimize(self._loglik, init_params, args=(y,), method='BFGS')
        self.params_ = res.x
        return self

    def predict(self, y, steps=1):
        omega = self.params_[0]
        alpha = self.params_[1:1+self.q]
        beta = self.params_[1+self.q:]
        
        eps = y
        n = len(eps)
        sigma2 = np.zeros(n+steps)
        start = max(self.p,self.q)
        sigma2[:start] = np.var(eps)
        
        for t in range(start, n):
            sigma2[t] = omega
            for i in range(self.q):
                sigma2[t] += alpha[i]*eps[t-1-i]**2
            for j in range(self.p):
                sigma2[t] += beta[j]*sigma2[t-1-j]

        forecasts = []
        # For forecasting we use the recursion forward assuming eps_t+future=0:
        for h in range(steps):
            t = n+h
            sigma2[t] = omega
            for i in range(self.q):
                # eps in future are zero
                sigma2[t] += alpha[i]*0  
            for j in range(self.p):
                sigma2[t] += beta[j]*sigma2[t-1-j]
            forecasts.append(sigma2[t])
        return np.array(forecasts)

class EGARCH:
    def __init__(self):
        # We'll implement a simple EGARCH(1,1)
        self.params_ = None  # [omega, alpha, gamma, beta]

    def _loglik(self, params, y):
        omega, alpha, gamma, beta = params
        eps = y
        n = len(y)
        sigma2 = np.ones(n)*np.var(y)
        
        for t in range(1, n):
            z = eps[t-1]/np.sqrt(sigma2[t-1])
            log_sigma2 = omega + alpha*z + gamma*(np.abs(z)-np.sqrt(2/np.pi)) + beta*np.log(sigma2[t-1])
            sigma2[t] = np.exp(log_sigma2)

        ll = -0.5*np.sum(np.log(2*np.pi) + np.log(sigma2) + eps**2/sigma2)
        return -ll

    def fit(self, y):
        init_params = np.array([0.0, 0.0, 0.0, 0.9])
        res = minimize(self._loglik, init_params, args=(y,), method='BFGS')
        self.params_ = res.x
        return self

    def predict(self, y, steps=1):
        omega, alpha, gamma, beta = self.params_
        eps = y
        n = len(y)
        sigma2 = np.ones(n+steps)*np.var(y)
        for t in range(1, n):
            z = eps[t-1]/np.sqrt(sigma2[t-1])
            log_sigma2 = omega + alpha*z + gamma*(np.abs(z)-np.sqrt(2/np.pi)) + beta*np.log(sigma2[t-1])
            sigma2[t] = np.exp(log_sigma2)

        forecasts = []
        # For forecasting, future eps=0
        current_sigma = sigma2[n-1]
        for h in range(steps):
            z = 0 # future shocks assumed zero
            log_sigma2 = omega + alpha*z + gamma*(np.abs(z)-np.sqrt(2/np.pi)) + beta*np.log(current_sigma)
            current_sigma = np.exp(log_sigma2)
            forecasts.append(current_sigma)
        return np.array(forecasts)

class TGARCH:
    def __init__(self):
        # We'll implement TARCH(1,1) only
        self.params_ = None # [omega, alpha, gamma, beta]

    def _loglik(self, params, y):
        omega, alpha, gamma, beta = params
        if omega <=0 or alpha<0 or beta<0 or (alpha+gamma)<0:
            return 1e9

        eps = y
        n = len(y)
        sigma2 = np.ones(n)*np.var(y)
        
        for t in range(1, n):
            I = 1 if eps[t-1]<0 else 0
            sigma2[t] = omega + alpha*(eps[t-1]**2) + gamma*(eps[t-1]**2)*I + beta*sigma2[t-1]

        ll = -0.5*np.sum(np.log(2*np.pi) + np.log(sigma2) + eps**2/sigma2)
        return -ll

    def fit(self, y):
        init_params = np.array([np.var(y)*0.1, 0.05, 0.05, 0.9])
        res = minimize(self._loglik, init_params, args=(y,), method='BFGS')
        self.params_ = res.x
        return self

    def predict(self, y, steps=1):
        omega, alpha, gamma, beta = self.params_
        eps = y
        n = len(eps)
        sigma2 = np.ones(n+steps)*np.var(y)
        for t in range(1, n):
            I = 1 if eps[t-1]<0 else 0
            sigma2[t] = omega + alpha*(eps[t-1]**2) + gamma*(eps[t-1]**2)*I + beta*sigma2[t-1]

        forecasts = []
        # future eps=0
        current_sigma = sigma2[n-1]
        for h in range(steps):
            I = 0 # no negative shocks in future
            current_sigma = omega + alpha*(0) + gamma*(0)*I + beta*current_sigma
            forecasts.append(current_sigma)
        return np.array(forecasts)

def priceChart(ticker, start="2010-01-01", end="2021-01-01"):
    # Fetch data
    data = yf.download(str(ticker), start=start, end=end)
    
    # Handle empty DataFrame
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker} between {start} and {end}")
    
    # Plot price chart
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label="Price", color="blue")
    plt.title(f"Price of {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    return plt

def logReturnsChart(ticker, start="2010-01-01", end="2021-01-01"):
    # Fetch data
    data = yf.download(ticker, start=start, end=end)
    
    # Handle empty DataFrame
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker} between {start} and {end}")
    
    # Calculate log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()  # Drop missing values
    
    # Plot log returns
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['log_return'], label="Log Returns", color="green")
    plt.title(f"Log Returns of {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend()
    plt.show()
    
    return plt
    
data = yf.download("AAPL", start="2020-01-01", end = "2021-01-01")
data['log_return'] = np.log(data['Close']/data['Close'].shift(1))
returns = data['log_return'].dropna().values

data = data.dropna()

# model = ARCH(p=1)
# model.fit(returns)

# plt.figure(figsize=(10, 6))
# plt.plot(data['log_return'], label="Log Returns")
# plt.title(f"Daily Log Returns of")
# plt.xlabel("Date")
# plt.ylabel("Log Return")
# plt.legend()
# plt.show()

# data['square_returns'] = data['log_return']**2

# plt.plot(data['square_returns'], label="Squared Log Returns")
# plt.title(f"Daily Log Returns of")
# plt.xlabel("Date")
# plt.ylabel("Squared Log Return")
# plt.legend()
# plt.show()