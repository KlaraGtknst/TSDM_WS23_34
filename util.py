import numpy as np
import matplotlib.pyplot as plt

def get_random_signal():
    Xnorm = np.random.multivariate_normal([0,0],[[.1,0],[0,10]], size=1000)
    Xnorm[354] = np.asarray([-20, 100])
    return Xnorm

def plot_equal(Xnorm):
    plt.figure()
    plt.axis('equal')
    plt.scatter(Xnorm[:,0], Xnorm[:,1])
    plt.show()
    
def plot_range(Xnorm, ax_range=(-1,1)):
    plt.figure()
    plt.xlim((ax_range))
    plt.ylim((ax_range))
    plt.scatter(Xnorm[:,0], Xnorm[:,1])
    plt.show()
    
def apply_plot(f, Xs, ax_range=(-1,1)):
    X = f(Xs)
    plot_range(X, ax_range)
    
def get_disturbed_signal():
    x = np.arange(0,30,.1)
    y = np.sin(x) + np.cos(3*x) + x*.05

    missing = np.random.choice(np.arange(len(y)), size=50)
    outlier = np.random.choice(np.arange(len(y)), size=5)

    y[missing] = np.nan
    y[outlier] = y[outlier] + 10 + 3 * np.random.uniform(len(outlier))
    return x,y

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        
        [src] https://stackoverflow.com/a/6520696
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def get_signal():
    # How many time points are needed i,e., Sampling Frequency
    samplingFrequency   = 100;

    # At what intervals time points are sampled
    samplingInterval       = 1 / samplingFrequency;

    # Begin time period of the signals
    beginTime           = 0;
    # End time period of the signals
    endTime             = 2; 

    # Frequency of the signals
    signal1Frequency     = 4;
    signal2Frequency     = 7;
    
    # Time points
    time = np.arange(beginTime, endTime, samplingInterval);

    # Create two sine waves
    amplitude1 = np.sin(2*np.pi*signal1Frequency*time)
    amplitude2 = np.sin(2*np.pi*signal2Frequency*time)
    
    # Add the sine waves
    amplitude = amplitude1 + amplitude2 + np.random.randn(len(amplitude1))*1
    
    # Time domain representation of the resultant sine wave
    plt.figure()
    plt.title('Sine wave with multiple frequencies')
    plt.plot(time, amplitude)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    
    return time, amplitude