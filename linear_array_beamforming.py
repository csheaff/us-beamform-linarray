
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal

def arange2(start, stop=None, step=1):
    """#Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop == None:
        a = np.arange(start)
    else: 
        a = np.arange(start, stop, step)
        if a[-1] > stop-step:   
            a = np.delete(a, -1)
    return a
    
sensorData = sio.loadmat('example_us_bmode_sensor_data.mat')['sensor_data'] #[sensorData] = 96x32x1585 -> transmission x recording element x time index

numTxBeams = sensorData.shape[0]
numProbeChan = sensorData.shape[1]
samplesPerAcq = sensorData.shape[2]

sampleRate = 27.72e6
toffset = 1.33e-6  #represents the time at which the middle of the transmission pulse occurs. Determined by inspection of signals
t = np.arange(samplesPerAcq)/sampleRate - toffset
txFreq = 1.5e6
c0 = 1540

## preprocessing 

n = 201
lc, hc = 0.5e6, 2.5e6
lc = lc/(sampleRate/2) #normalize to nyquist frequency
hc = hc/(sampleRate/2)
B = signal.firwin(n, [lc, hc], pass_zero=False) #band-pass filter

interpFact = 4
sampleRate = sampleRate*interpFact
samplesPerAcq2 = samplesPerAcq*interpFact
apodWin = signal.tukey(numProbeChan)

dataApod = np.zeros((numTxBeams, numProbeChan, samplesPerAcq2))
for m in range(numTxBeams):
    for n in range(numProbeChan):
        w = sensorData[m,n,:]
        if np.sum(w) != 0:
            dataFilt = signal.lfilter(B, 1, w)
            dataInterp = signal.resample_poly(dataFilt, interpFact, 1)
            dataApod[m,n,:] = apodWin[n]*dataInterp

freqs, delay = signal.group_delay((B,1))
delay = int(delay[0])*interpFact
t2 = np.interp(arange2(0,len(t),1/interpFact), np.arange(len(t)), t)-delay/sampleRate

## remove signal before t = 0

f = np.where(t2 < 0)[0]
t2 = np.delete(t2, f)
dataApod = dataApod[:,:,f[-1]+1:]
samplesPerAcq = dataApod.shape[2]

## beamforming
res = 10e-6
zd = arange2(2.5e-3, 40e-3, res)
zd2 = zd**2

transPitch = 2*1.8519e-4
xd = np.arange(numProbeChan)*transPitch
xd = xd - np.max(xd)/2 #transducer locations

propDist = np.zeros((numProbeChan, len(zd)))
for r in range(numProbeChan):
    dist1 = zd
    dist2 = np.sqrt(xd[r]**2+zd2)
    propDist[r,:] = dist1 + dist2
propDistInd = np.round(propDist/c0*sampleRate)
propDistInd = propDistInd.astype('int')

tgc_alpha = 0.25 #units of dB/MHz-cm
tgc = np.exp(tgc_alpha*txFreq/1e6*2*zd*100)

scanLine = np.zeros(len(zd))
image = np.zeros((numTxBeams, len(zd)))
for q in range(numTxBeams):
    for r in range(numProbeChan):
        v = dataApod[q,r,:]
        scanLine = scanLine + v[propDistInd[r,:]]
    env = np.abs(signal.hilbert(scanLine))
    envAmp = env*tgc
    image[q,:] = envAmp
    scanLine = np.zeros(len(zd))

imageLog = 20*np.log10(image/np.max(image))
dr = 30

xd2 = np.arange(numTxBeams)*transPitch
xd2 = xd2 - np.max(xd2)/2

fig1 = plt.figure()
plt.imshow(np.transpose(imageLog), extent=[xd2[0]*1e3,xd2[-1]*1e3,zd[-1]*1e3,zd[0]*1e3], vmin=-dr, vmax=0, cmap='gray')
plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.colorbar()
plt.show()
