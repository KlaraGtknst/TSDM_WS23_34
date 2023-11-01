import numpy as np
import matplotlib.pyplot as plt

def TS_1(x1=20, samples=300, with_trend=True):
    """
    Returns:
    ========
    x,y - timestamps and values of time series for SAX exercise.
    """
    x = np.linspace(0, x1, samples)

    trend = lambda x : 0.05*x+1
    np.random.seed(5000)
    y = 0.5*np.sin(x) + np.random.uniform(size=len(x)) + (trend(x) if with_trend else 0)
    return x,y

def TS_2(x1=20, samples=300, with_trend=True):
    """
    Returns:
    ========
    x,y - timestamps and values of time series for SAX exercise.
    """
    x = np.linspace(0, x1, samples)

    trend = lambda x : 0.02*x+1
    np.random.seed(5000)
    y = 0.5*np.cos(x) + np.random.uniform(size=len(x)) + (trend(x) if with_trend else 0)
    return x,y
	
def ppoly_data():
	"""
	if x < -10, then y = -3 · x + 4

	if -10 < x < 15, then y = x + 44

	if x > 15, then y = -4 · x + 119
	"""
	np.random.seed(9999)
	x = np.random.normal(0, 1, 1000) * 10
	y = np.where(x < -10, -3 * x + 4 , np.where(x < 15, x + 44, -4 * x + 119)) + np.random.normal(0, 3, 1000)

	return x, y