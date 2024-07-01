import numpy as np
import pandas as pd

# ======================================
# SECTION: VIMD to VIMFC
# ======================================

vimd = np.loadtxt('LVIMD.csv')  # Data "time series" sebagai input
vimfc = -vimd
print(vimfc)
##DONE!! np.savetxt('VIMFCL.csv', vimfc, delimiter=',', header='VIMFC', comments='')

# ======================================
# SECTION: Read SST Dats
# ======================================
sst = np.loadtxt('sstbopung.csv')
sst_s = pd.Series(sst)
sst_a = sst_s - np.mean(sst_s)

# ======================================
# SECTION: ONI Index from VIMFC and SST
# ======================================
oni = sst_a.rolling(window = 3).mean()
## DONE!! oni.to_csv('ONI.csv')
O = np.loadtxt('ONI.csv')

# ======================================
# SECTION: WAVELET ONI INDEX
# ======================================

import pywt
import pywt.data
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

dt = 1 #1 hour
time = np.arange(len(vimfc)) * dt + 0
xlim = ([1,96])

# --- Plot time series
fig = plt.figure(figsize=(9, 10))
gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                    wspace=0, hspace=0)
plt.subplot(gs[0, 0:3])
plt.plot(time, vimfc, 'k', label='VIMFC')
plt.plot(time, O, 'r', label='ONI Index')  # New line in red dashed style
plt.xlim(xlim[:])
plt.xlabel('Time (hour)')
plt.ylabel('NINO3.4 ONI Index (\u00B0C)')
plt.title('a) NINO3.4 ONI Index (30 November - 3 December 2015)')
plt.legend()
plt.grid(True)

plt.text(time[-1] + 35, 0.5, 'Wavelet Analysis\nC. Torrence & G.P. Compo\n'
    'http://paos.colorado.edu/\nresearch/wavelets/',
    horizontalalignment='center', verticalalignment='center')

plt.show()
