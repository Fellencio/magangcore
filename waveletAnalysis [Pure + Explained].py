import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np

# from mpl_toolkits.axes_grid1 import make_axes_locatable

from waveletFunctions import wave_signif, wavelet

__author__ = 'Aufaa Ikram Ramadika'


# WAVETEST Example Python script for WAVELET, using NINO3 SST dataset

# See "http://paos.colorado.edu/research/wavelets/"
# The Matlab code written January 1998 by C. Torrence
# modified to Python by Evgeniya Predybaylo, December 2014

# Modified Oct 1999, changed Global Wavelet Spectrum (GWS) to be sideways,
# changed all "log" to "log2", changed logarithmic axis on GWS to
# a normal axis.

# ======================================
# SECTION: MEMBACA DATA
# ======================================

sst = np.loadtxt('historical_weather_data.csv')  # Data "time series" SST sebagai input
sst = sst - np.mean(sst) # Untuk mencari SST anomaly menggunakan rumus ini
variance = np.std(sst, ddof=1) ** 2
## np: This refers to the NumPy library, which is commonly used for numerical operations in Python.
## std: This is a function in NumPy that calculates the standard deviation.
## sst: This is the array (or list) of data points for which you want to calculate the standard deviation.
## ddof=1: This stands for "Delta Degrees of Freedom". When calculating the standard deviation, setting ddof=1 means you're using the sample standard deviation formula, which divides by 
## 𝑛−1 (where n is the number of data points) instead of n. This is also known as Bessel's correction and is used when you're estimating the standard deviation from a sample rather than a full population.
## ** 2: This part squares the result of the standard deviation calculation.

print("variance = ", variance) # untuk menunjukkan nilai var di code

# ======================================
# SECTION: KOMPUTASI
# ======================================

if 0: 
    variance = 1.0
    sst = sst / np.std(sst, ddof=1)
# Above is a conditional statement that will never execute because 0 is equivalent to False in Python.

n = len(sst) #calculates the length of the array 'sst' and stores it in the variable 'n'
dt = 0.25
# represent the time interval between data points in the 'sst' array
# the time step 'dt' should match the interval at which your data was collected
# if your data points are collected every hour, 'dt' would be 1 hour

time = np.arange(len(sst)) * dt + 1871.0
# 1 line above is the code to create a time array starting from 1871
# 'np.arrange(len(sst))' generates an array of integers from 0 to 'len(sst)-1'
# multiplying by 'dt' scales this array by the time step, and adding 1871 shifts the entire array to start from 1871

xlim = ([1870, 2000])  # plotting range
pad = 1  # pad the time series with zeroes (recommended)
dj = 0.25
# spacing between discrete scales in a wavelet trans, this will do 4 sub-octaves per octave
# larger values mean fewer scales between octaves, leading to a coarser analysis but less computational load

s0 = 2 * dt
# This sets the starting scale s0 to twice the time step dt. If dt is 0.25, then s0 will be 0.5.
# This could indicate the smallest scale of analysis in a wavelet transform,
# corresponding to 6 months if dt is in years.

j1 = 7 / dj
# This could represent the range of scales in a wavelet transform,
# covering 7 octaves with 4 sub-octaves each.

lag1 = 0.72  # lag-1 autocorrelation for red noise background, use excel to know
print("lag1 = ", lag1)
mother = 'MORLET'

# ======================================
# SUB-SECTION: WAVELET TRANSFORMATION
# ======================================

wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
## pad: A boolean value indicating whether to pad the time series data with zeroes to the next power of 2.
## wave: The complex wavelet coefficients.
## period: The vector of Fourier periods corresponding to the scales.
## scale: The vector of wavelet scales.
## coi: The cone of influence, which indicates the region of the wavelet power spectrum where edge effects become significant.
power = (np.abs(wave)) ** 2
## This power spectrum gives a measure of the variance distribution of the time series data in the time-frequency domain.
global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
##The global wavelet spectrum provides a measure of the average variance distribution across scales (or periods) for the entire time series.

### In summary, this code applies the wavelet transform to a time series dataset to obtain the wavelet power spectrum and then computes the global wavelet spectrum by averaging the power spectrum
### over time. This analysis helps in understanding the dominant modes of variability and how they change over time and across different scales.

# Significance levels:
signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
    lag1=lag1, mother=mother)
## sigtest=0: This indicates a significance test against a background red noise spectrum.
## The wave_signif function returns the significance levels of the wavelet power spectrum.
# expand signif --> (J+1)x(N) array
sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
sig95 = power / sig95  # where ratio > 1, power is significant
## signif[:, np.newaxis]: Converts the 1D array signif to a 2D array with shape (J+1, 1).
## np.ones(n)[np.newaxis, :]: Creates a 2D array of ones with shape (1, N).
## sig95: The result is a 2D array with shape (J+1, N) representing the significance levels expanded across all time points.
## power / sig95: Where the ratio is greater than 1, the power is considered significant at the 95% level.

# Global wavelet spectrum & significance levels:
dof = n - scale  # the -scale corrects for padding at edges
global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, #sigtest=1 indicates sigtest for global wavelet spectrum
    lag1=lag1, dof=dof, mother=mother)

# Scale-average between El Nino periods of 2--8 years
avg = np.logical_and(scale >= 2, scale < 8) #Creates a boolean mask for scales between 2 and 8 years (El Niño periods)
Cdelta = 0.776  # this is for the MORLET wavelet
# expand scale --> (J+1)x(N) array
scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :]) #Expands the scale array to (J+1,N)
scale_avg = power / scale_avg  #Normalizes the power by the scale
scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  #Computes the scale-averaged wavelet power spectrum for the specified period
scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
    lag1=lag1, dof=([2, 7.9]), mother=mother) #Degrees of freedom for the scale-averaged significance test, specified for the 2-8 year period.

## ======================================
## SUB-SUB-SECTION: Aufaa's Curiosity
## ======================================

#=# How do we create a scale-average of the wavelet power spectrum for monthly/daily/hourly period

# Let's say we're interested in periods between 2 to 5 days
min_period_hours = 2 * 24 # 2 days in hours
max_period_hours = 5 *24 # 5 days in hours

# Assume 'sst' is the time series data and dt = 1, then we apply the wavelet transform
wave, period, scale, coi = wavelet(sst, dt=1, pad=pad, dj=dj, s0=s0, j1=j1, mother=mother)
power = (np.abs(wave)) ** 2

# Then we compute the significance levels
signif = wave_signif([variance], dt=1, sigtest=0, scale=scale, lag1=lag1, mother=mother)
sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
sig95 = power / sig95

# Then we perform the scale-averaging
avg = np.logical_and(period >= min_period_hours, period < max_period_hours)
Cdelta = 0.776  # for Morlet wavelet
scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
scale_avg = power / scale_avg  # Normalize by scale
scale_avg = dj * dt / Cdelta * np.sum(scale_avg[avg, :], axis=0)  # Scale-average

# Lastly, calculate the significance for the scale-averaged power spectrum
dof = n - scale  # Degrees of freedom, corrected for edge effects
scaleavg_signif = wave_signif(variance, dt=1, scale=scale, sigtest=2, lag1=lag1, dof=([min_period_hours, max_period_hours - 1]), mother=mother)

#=# How to add another line to the same plot

plt.plot(time, vimfc, 'k', label='VIMFC')
plt.plot(time, O, 'r', label='ONI Index')  # New line in red dashed style
#You can see each colors and line styles on matplotlib

# ======================================
# SUB-SECTION: PLOTTING
# ======================================

# --- Plot time series
fig = plt.figure(figsize=(9, 10))
## This creates a new figure object with a specified size of 9 inches by 10 inches.
gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
## This creates a grid specification (GridSpec) for the figure with 3 rows and 4 columns, and sets the horizontal and vertical (hspace) space between the subplots.
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                    wspace=0, hspace=0)
## This function fine-tunes the spacing of the subplots within the figure. It sets the left, bottom, right, and top margins, as well as the width (wspace) and height (hspace) spaces between the subplots.
plt.subplot(gs[0, 0:3])
## This creates a subplot in the specified location within the grid. Here, it spans from the first row (0) and covers the first three columns (0:3).
plt.plot(time, sst, 'k')
## This plots the sst (Sea Surface Temperature) data against time with a black line ('k' stands for black).
plt.xlim(xlim[:])
plt.xlabel('Time (year)')
plt.ylabel('NINO3 SST (\u00B0C)')
plt.title('a) NINO3 Sea Surface Temperature (seasonal)')

plt.text(time[-1] + 35, 0.5, 'Wavelet Analysis\nC. Torrence & G.P. Compo\n'
    'http://paos.colorado.edu/\nresearch/wavelets/',
    horizontalalignment='center', verticalalignment='center')
## time[-1] + 35 sets the x-coordinate to 35 units past the last time value
## 0.5 sets the y-coordinate to 0.5

# --- Contour plot wavelet power spectrum
# plt3 = plt.subplot(3, 1, 2)
plt3 = plt.subplot(gs[1, 0:3])
## Creates a subplot in the 2nd row (1 is 2nd) and spanning the first 3 columns for the grid gs
levels = [0, 0.5, 1, 2, 4, 999]
## to define the levels for the contour plot
# *** or use 'contour'
CS = plt.contourf(time, period, power, len(levels))
## It creates a filled contour plot of the wavelet power spectrum with the number of levels specified
## The period refers to the time scale of the oscillations present in the data, represents the duration of one complete cycle of the wavelet at a particular scale
im = plt.contourf(CS, levels=levels,
    colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
plt.xlabel('Time (year)')
plt.ylabel('Period (years)')
plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4\u00B0C$^2$)')
plt.xlim(xlim[:])
# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
plt.contour(time, period, sig95, [-99, 1], colors='k')
## Adds a contour for the 95% significance level. The levels [-99, 1] include a fake level (-99) to properly display the significant areas.
# cone-of-influence, anything "below" is dubious
plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
    edgecolor="#00000040", hatch='x') ##  to indicate regions below the cone of influence (coi) where edge effects might distort the analysis.
plt.plot(time, coi, 'k') # format y-scale
plt3.set_yscale('log', base=2, subs=None) ## to format the y-axis and set it to a lof scale with base 2
plt.ylim([np.min(period), np.max(period)]) ## Sets the limits of the y-axis to the range of the periods.
ax = plt.gca().yaxis ## gets the current y-axis
ax.set_major_formatter(ticker.ScalarFormatter()) ## Sets the y-axis formatter to a scalar formatter to avoid scientific notation.
plt3.ticklabel_format(axis='y', style='plain') ## Ensures the y-axis tick labels are in plain style.
plt3.invert_yaxis() ## Inverts the y-axis to have the larger periods at the bottom and smaller periods at the top.
# set up the size and location of the colorbar
# position=fig.add_axes([0.5,0.36,0.2,0.01])
# plt.colorbar(im, cax=position, orientation='horizontal')
#   , fraction=0.05, pad=0.5)

# plt.subplots_adjust(right=0.7, top=0.9)

# --- Plot global wavelet spectrum
plt4 = plt.subplot(gs[1, -1]) ##2nd row last column
plt.plot(global_ws, period) ##plots it against the period
plt.plot(global_signif, period, '--') ##plots it against the period using a dashed line
plt.xlabel('Power (\u00B0C$^2$)')
plt.title('c) Global Wavelet Spectrum')
plt.xlim([0, 1.25 * np.max(global_ws)])
##Sets the limits of the x-axis from 0 to 1.25 times the maximum value of global_ws to provide some padding on the right side.
# format y-scale
plt4.set_yscale('log', base=2, subs=None)
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt4.ticklabel_format(axis='y', style='plain')
plt4.invert_yaxis()

# --- Plot 2--8 yr scale-average time series
plt.subplot(gs[2, 0:3]) ##3rd row 
plt.plot(time, scale_avg, 'k')
plt.xlim(xlim[:])
plt.xlabel('Time (year)')
plt.ylabel('Avg variance (\u00B0C$^2$)')
plt.title('d) 2-8 yr Scale-average Time Series')
plt.plot(xlim, scaleavg_signif + [0, 0], '--')

plt.show()
