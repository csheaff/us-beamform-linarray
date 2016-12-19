import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal

# global constants
numTxBeams = 96
numProbeChan = 32
txFreq = 1.5e6
c0 = 1540
transPitch = 2*1.8519e-4
sampleRate = 27.72e6


def arange2(start, stop=None, step=1):
    """#Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop == None:
        a = np.arange(start)
    else: 
        a = np.arange(start, stop, step)
        if a[-1] > stop-step:   
            a = np.delete(a, -1)
    return a

def getTGC(alpha0, propDist):
    """ Time-gain compensation
    Model for amplitude reduction with distance -> A(z) = A(0)*exp(-mu*z)
    It's convenient to think of A(z)/A(0) in dB, i.e. 20*log10(A(z)/A(0))
    which happens to = 20*log10(exp(-mu*z)) = 20*log10(e)*(-mu*z) ~ -8.7*mu*z
    As the attentuation coefficient, it's common to specify the quantity 8.7*mu and call it alpha
    Because attentuation is frequency-dependent, alpha is additionally expressed as alpha = alpha0*f^n (f = freq, 1<n<2)
    and the value alpha0 is usually given for a tissue type in units of dB/(MHz-cm).
    Therefore, A(z)/A(0) = exp(-mu*z) = exp(-alpha/8.7*z) = exp(-alpha0*f^n/8.7*z)
    To compensate for attentuation, we can therefore multiple by exp(alpha0*f^n/8.7*z).
    However, users will usually customize TGC settings using external controls, as they tend to like
    their levels at particular settings which aid their diagnosis."""
    #alpha0 = 0.4;   # [a0] = dB/(MHz-cm),  0.54 is average for soft tissue
    n = 1;  # approx. 1 for soft tissue
    alpha = alpha0*(txFreq*1e-6)**n;  
    mu = alpha/8.7 # convert out of dB units
    tgcGain = np.exp(mu*propDist*1e2);  # double zd for round-trip distance, convert to cm
    return tgcGain

def envDet(scanLine, t, method='hilbert'):
    """Envelope detection. This can be done in a few ways: 
    (1) Hilbert transform method
        - doesn't require knowledge of carrier frequency
        - simple - doesn't require filtering
        - cannot be implement with analog electronics
        - edge effects are undesirable
      
    (2) Demodulation + Low-pass filtering
        - implementable with analog electronics
        - requires knowledge of the carrier frequency, which gets smaller with propagation
        - more computational steps involved.

    'demod' and 'demod2' do exactly the same thing here. The former is merely the simplest/most intuitive 
    way to look at the operation (multiplying by complex exponential yields a frequency shift in the fourier domain).
    Whereas with the latter, the I and Q components are defined, as is typical. 
    """
    n = 201
    fs = 1/(t[1]-t[0])
    lc = 0.75e6
    b = signal.firwin(n, lc/(fs/2))  #low-pass filter

    if method == 'hilbert':
        envelope = np.abs(signal.hilbert(scanLine))
    elif method == 'demod':
        demodulated = scanLine*np.exp(-1j*2*np.pi*txFreq*t)
        demodFilt = np.sqrt(2)*signal.filtfilt(b, 1, demodulated)  #using zero-phase filter to avoid time delay
        envelope = np.abs(demodFilt)
    elif method == 'demod2':
        I = scanLine*np.cos(2*np.pi*txFreq*t)
        If = np.sqrt(2)*signal.filtfilt(b, 1, I)
        Q = scanLine*np.sin(2*np.pi*txFreq*t)
        Qf = np.sqrt(2)*signal.filtfilt(b, 1, Q)
        envelope = np.sqrt(If**2+Qf**2)        
    return envelope

def preprocUS(data, t, xd):
    sampleRate = 1/(t[1]-t[0])
    samplesPerAcq = data.shape[2]
    
    a0 = 0.4
    
    # get time-gain compensation vectors based on estimate for propagation distance to each element
    zd = t*c0/2
    zd2 = zd**2
    dist1 = zd
    tgc = np.zeros((numProbeChan,samplesPerAcq))
    for r in range(numProbeChan):
        dist2 = np.sqrt(xd[r]**2+zd2)
        propDist = dist1 + dist2
        tgc[r,:] = getTGC(a0, propDist) 

    # apply tgc 
    dataAmp = np.zeros(data.shape)     
    for m in range(numTxBeams):
        dataAmp[m,:,:] = data[m,:,:]*tgc
            
    # retrieve filter coefficients
    filtOrd = 201
    lc, hc = 0.5e6, 2.5e6
    lc = lc/(sampleRate/2) #normalize to nyquist frequency
    hc = hc/(sampleRate/2)
    B = signal.firwin(filtOrd, [lc, hc], pass_zero=False) #band-pass filter

    # specify interpolation factor, get apodizatin window
    interpFact = 4
    sampleRate = sampleRate*interpFact
    samplesPerAcq2 = samplesPerAcq*interpFact
    apodWin = np.ones(numProbeChan) #signal.tukey(numProbeChan)

    dataApod = np.zeros((numTxBeams, numProbeChan, samplesPerAcq2))
    for m in range(numTxBeams):
        for n in range(numProbeChan):
            w = dataAmp[m,n,:]
            if np.sum(w) != 0:
                dataFilt = signal.lfilter(B, 1, w)
                dataInterp = signal.resample_poly(dataFilt, interpFact, 1)
                dataApod[m,n,:] = apodWin[n]*dataInterp

    # create new time vector based on interpolation and filter delay
    freqs, delay = signal.group_delay((B,1))
    delay = int(delay[0])*interpFact
    t2 = np.interp(arange2(0,len(t),1/interpFact), np.arange(len(t)), t)-delay/sampleRate

    # remove signal before t = 0
    f = np.where(t2 < 0)[0]
    t2 = np.delete(t2, f)
    dataApod = dataApod[:,:,f[-1]+1:]

    return dataApod, t2, tgc


def beamform(data, t, xd, receiveFocus):
    Rf = receiveFocus
    fs = 1/(t[1]-t[0])
    delayInd = np.zeros(numProbeChan, dtype=int)
    for r in range(numProbeChan):
        delay = Rf/c0*(np.sqrt((xd[r]/Rf)**2+1)-1)
        delayInd[r] = int(round(delay*fs))
    maxDelay = np.max(delayInd)
    
    waveformLength = data.shape[2]
    image = np.zeros((numTxBeams,waveformLength)) #initialize
    for q in range(numTxBeams):
        scanLine = np.zeros(waveformLength + maxDelay) #initialize
        for r in range(numProbeChan):
            delayPad = np.zeros(delayInd[r])
            fillPad = np.zeros(len(scanLine)-waveformLength-delayInd[r])
            waveform = data[q,r,:]
            scanLine = scanLine + np.concatenate((fillPad, waveform, delayPad))
        image[q,:] = scanLine[maxDelay:]
    z = t*c0/2
    return image, z
        
def beamformDF(data, t, xd):

    sampleRate = 1/(t[2]-t[1])
    
    zd = t*c0/2
    zd2 = zd**2
    propDist = np.zeros((numProbeChan, len(zd)))
    for r in range(numProbeChan):
        dist1 = zd
        dist2 = np.sqrt(xd[r]**2+zd2)
        propDist[r,:] = dist1 + dist2
    propDistInd = np.round(propDist/c0*sampleRate)
    propDistInd = propDistInd.astype('int')  #acoustic propagation distance from transmission to reception for each element
                                         #these distances stay the same as we slide across the aperture of the full array

    f = np.where(propDistInd >= len(t))  # eliminate indices that are out of bounds
    propDistInd[f[0],f[1]] = len(t)-1  # replace out-of-bound indices with last index (likely to be of low signal 
                                       # at that location i.e closest to a null
    scanLine = np.zeros(len(zd))
    image = np.zeros((numTxBeams, len(zd)))
    for q in range(numTxBeams):  #index transmission
        for r in range(numProbeChan):  #index receiver
            v = data[q,r,:]      #get recorded waveform
            scanLine = scanLine + v[propDistInd[r,:]]  #index waveform at times corresponding to propagation distance to pixel along a-line
        image[q,:] = scanLine
        scanLine = np.zeros(len(zd))
    return image, zd, propDistInd

#def main():

# load data from file
sensorData = sio.loadmat('example_us_bmode_sensor_data.mat')['sensor_data'] #[sensorData] = 96x32x1585 -> transmission x recording element x time index

# data get info
samplesPerAcq = sensorData.shape[2]

toffset = 1.33e-6  #represents the time at which the middle of the transmission pulse occurs. Determined by inspection of signals
t = np.arange(samplesPerAcq)/sampleRate - toffset

xd = np.arange(numProbeChan)*transPitch
xd = xd - np.max(xd)/2 #transducer locations relative to the a-line, which is always centered

# preprocessing - signal filtering, interpolation, and apodization
dataApod, t2, tgc = preprocUS(sensorData, t, xd) 
        
# beamforming with dynamic focusing
imageDF, zDF, propdist = beamformDF(dataApod, t2, xd)

rxFocus = 35e-3
image, z = beamform(dataApod, t2, xd, rxFocus)

im = imageDF
Z = zDF

# nullify beginning of image that includes transmission pulse

f = np.where(Z < 5e-3)[0]
ZTrunc = np.delete(Z,f)
imTrunc = im[:,f[-1]+1:]

# envelope detection
for n in range(numTxBeams):
    imTrunc[n,:] = envDet(imTrunc[n,:], 2*ZTrunc/c0 , method = 'hilbert')         #and add contributions across all 32 channels
    
# apply time-gain compensation
#a0 = 0.4
#tgcGain = getTGC(a0, zd, txFreq)
#for n in range(numTxBeams):
#    imageDF[n,:] = imageDF[n,:]*tgcGain
            
# log compression and scan conversion

imageLog = 20*np.log10(imTrunc/np.max(imTrunc))
dr = 30
        
xd2 = np.arange(numTxBeams)*transPitch
xd2 = xd2 - np.max(xd2)/2

# plotting

fig1 = plt.figure()
#plt.imshow(np.transpose(imageLog),vmin=-dr, vmax=0, cmap = 'gray', aspect = 'auto')
plt.imshow(np.transpose(imageLog), extent=[xd2[0]*1e3,xd2[-1]*1e3,ZTrunc[-1]*1e3,ZTrunc[0]*1e3], vmin=-dr, vmax=0, cmap='gray')
plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.colorbar()
plt.show()


#if __name__ == '__main__':
 #   main()





