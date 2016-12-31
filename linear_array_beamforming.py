import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp2d

# global constants
numTxBeams = 96
numProbeChan = 32
txFreq = 1.5e6
txFocus = 20e-3
c0 = 1540
transPitch = 2*1.8519e-4
sampleRate = 27.72e6
toffset = 1.33e-6  #time at which the middle of the transmission pulse occurs

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
    The attenuation coefficient of tissue is usually expressed in dB and defined as
    alphaDB = 20/x*log10[p(0)/p(x)]
    where x is the propagation distance, p(0) is the incident pressure, p(x) is the spatially variant pressure
    As a result,  p(x)/p(0) = 10^(-alphadB*x/20)

    Additionally, alpha is modeled as alphaDB = alpha0*f^n, where f is frequency and 0 < n < 1
    For tissue, n ~ 1. alpha0 is usually the parameter specified in units of dB/(MHz-cm). We can compensate
    therefore by multiplying each A-line by 10^(alpha0*f*propDist*100/20). Note that this does not take into
    account the dissipation of acoustic energy with distance due to non-plane wave propagation.

    inputs:  alpha0 - attenutation coefficient in dB/(MHz-cm)
             propDist - round-trip propagation distance of acoustic pulse in meters

    outputs: tgcGain - gain vector for multiplication with A-line """

    n = 1  # approx. 1 for soft tissue
    alpha = alpha0*(txFreq*1e-6)**n;  
    tgcGain = 10**(alpha*propDist*100/20)

    return tgcGain

def preprocUS(data, t, xd):
    """Analog time-gain compensation is typically applied followed by an anti-aliasing filter (low-pass) and then A/D conversion. 
    The input data is already digitized here, so no need for anti-alias filtering. Following A/D conversion, one would ideally begin beamforming, however
    the summing process in beamforming can produce very high values if low frequencies are included. This can result in the generation
    of a dynamic range in the data that exceeds what's allowable by the number of bits, thereby yielding 
    data loss. Therefore it's necessary to high-pass filter before beamforming. In addition, beamforming is 
    more accurate with a higher sampling rate because the calculated beamforming delays are more accurately 
    achieved. Hence interpolation is used to upsample the signal. Finally, apodization is applied before the beamformer.
    
    This preprocessing function therefore consists of:
    1) time-gain compensation
    2) filtering
    3) interpolation
    4) apodization 
    
    In the filtering step I've appied a band-pass, as higher frequencies are also problematic and are usually
    addressed after beamforming. 
    
    inputs: data - transmission number x receive channel x time index
            t - time vector [s]
            xd - dector position vector [m]
    
    outputs: dataApod - processed data
             t2 - new time vectorssss    
    """

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

    # specify interpolation factor 
    interpFact = 4
    sampleRate = sampleRate*interpFact
    samplesPerAcq2 = samplesPerAcq*interpFact

    #get apodization window
    apodWin =  signal.tukey(numProbeChan) #np.ones(numProbeChan)

    #process
    dataApod = np.zeros((numTxBeams, numProbeChan, samplesPerAcq2))
    for m in range(numTxBeams):
        for n in range(numProbeChan):
            w = dataAmp[m,n,:]
            dataFilt = signal.lfilter(B, 1, w)
            dataInterp = signal.resample_poly(dataFilt, interpFact, 1)
            dataApod[m,n,:] = apodWin[n]*dataInterp

    # create new time vector based on interpolation and filter delay
    freqs, delay = signal.group_delay((B,1))
    delay = int(delay[0])*interpFact
    t2 = np.arange(samplesPerAcq2)/sampleRate + t[0] - delay/sampleRate
    
    #remove signal before t = 0
    f = np.where(t2 < 0)[0]
    t2 = np.delete(t2, f)
    dataApod = dataApod[:,:,f[-1]+1:]

    return dataApod, t2


def beamform(data, t, xd, receiveFocus):
    """This employs the classic delay-and-sum method of beamforming entailing a single focus location
    defined by receiveFocus [m]. 
    inputs: 
            data - RF data, dimenions of (transmission number, receive channel, time index)
            t - time vector associated with RF waveforms, [t] = seconds
            xd - horizontal position vector of receive channels relative to center, [xd] = meters
            receiveFocus - depth of focus for beamforming, [receiveFocus] = meters
    outputs: 
            image - beamformed data, dimensions of (scanline index, depth)
    """
    Rf = receiveFocus
    fs = 1/(t[1]-t[0])
    delayInd = np.zeros(numProbeChan, dtype=int)
    for r in range(numProbeChan):
        delay = Rf/c0*(np.sqrt((xd[r]/Rf)**2+1)-1)   #represents difference between propagation time for a central element
        delayInd[r] = int(round(delay*fs))           #and prop time for a off-centered element
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
    return image
        
def beamformDF(data, t, xd):
    """Ideally we could focus at all depths in receive when beamforming. This is done in an FPGA by using time delays
    that are time-varying. To clarify, suppose we use the above beamform function to focus at some depth z0. Why not use the delay
    to achieve this focus merely for the value at that depth? For some depth z0+dz, we would then have a new delay and use it to 
    generate the pixel only at z0+dz. So an array of time-dependent delay values can be generated for each channel that would allow 
    focusing at each depth. 

    In order to achieve dynamic focusing offline, digitally, one could find the time-dependent delays and apply them, but this would 
    require operating a loop over each time value. One could also use the above beamform function for each focal point and only keep
    the value generated for that depth, but again this would computationally wasteful. An alternative is to fill the a-line first with 
    values corresponding to the propagation time from emmission to pixel to receiver. One can then simply index the signal 
    received by an element at the estimtae for propagation time and add that to the pixel, followed by summing contributions from other 
    channels. Focusing at all depths is effectively acheived, and this is the method applied below.
    
    inputs: 
            data - RF data, dimenions of (transmission number, receive channel, time index)
            t - time vector associated with RF waveforms, [t] = seconds
            xd - horizontal position vector of receive channels relative to center, [xd] = meters
            
    outputs: 
            image - beamformed data, dimensions of (scanline index, depth)

"""
    sampleRate = 1/(t[2]-t[1])
    
    zd = t*c0/2  #note we can actually define this arbitrarily to get a higher resolution. I've refrained from doing this in
    zd2 = zd**2  #order to have a fair comparison to the non dynamic focusing method.
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
    return image

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

def logCompress(data, dynamicRange, rejectLevel, brightGain):
    """Dynamic range is defined as the max value of some data divided by the minimum value, and it is a measure of 
    how spread out the data values are. If the data values have been converted to dB, then dynamic range is defined
    as the max value minus the minimum value. 

    One could interpret there being two stages of compression in the standard log compression process. The first is 
    the simple conversion to dB. The second is in selecting to display a certain range of dB. 

    inputs:
            data - envelope-detected data having values >= 0. Dimensions should be scanline x depth/time index
            dynamicRange - desired dynamic range of data to present [dB]
            rejectLevel - level of rejection [dB]
            brightGain - brightness gain [dB]
    output:
            xdB3 - processed image, dimensions of scanline x depth/time index
    """

    #compress to dynamic range chosen
    xdB = 20*np.log10(1+data) #add one b/c log10(0) = -inf ('data' should have values >= 0)
    xdB2 = xdB - np.max(xdB) #shift such that max is 0 dB
    xdB3 = xdB2 + dynamicRange #shift such that max is dynamicRange value
    xdB3[np.where(xdB3 < 0)] = 0 #eliminate data outside of dynamic range

    #rejection
    xdB3[np.where(xdB3 <= rejectLevel)] = 0

    #add brightness gain
    xdB3 = xdB3 + brightGain
    xdB3[np.where(xdB3 > dynamicRange)] = dynamicRange  #keep maximum value equal to dynamicRange
    
    return xdB3

def scanConv(data, xb, zb):
    """create 512x512 pixel image
    inputs: data - scanline x depth/time
            xb - horizontal distance vector
            zb - depth vector
    outputs: imageSC - scanline x depth/time
             znew - new depth vector
             xnew - new horizontal distance vector"""

    interpFunc = interp2d(zb, xb, data, kind='linear')
    xnew = np.linspace(np.min(xb),np.max(xb), 512)
    znew = np.linspace(np.min(zb),np.max(zb), 512)
    imageSC = interpFunc(znew, xnew)

    return imageSC, znew, xnew
    
def main():

    # load data from file
    sensorData = sio.loadmat('example_us_bmode_sensor_data.mat')['sensor_data'] #[sensorData] = 96x32x1585 -> transmission x recording element x time index

    # data get info
    samplesPerAcq = sensorData.shape[2]

    # time vector for data
    t = np.arange(samplesPerAcq)/sampleRate - toffset

    xd = np.arange(numProbeChan)*transPitch
    xd = xd - np.max(xd)/2 #transducer locations relative to the a-line, which is always centered

    # preprocessing - signal filtering, interpolation, and apodization
    dataApod, t2 = preprocUS(sensorData, t, xd) 

    # simple B-mode image - no beamforming (only use waveform from a central array element)
    image = dataApod[:,15,:]  
    
    # beamforming with different receive focii
    rxFocus = 15e-3
    imageBF1 = beamform(dataApod, t2, xd, rxFocus)
    
    rxFocus = 35e-3
    imageBF2 = beamform(dataApod, t2, xd, rxFocus)

    # beamforming with dynamic focusing
    imageDF = beamformDF(dataApod, t2, xd)
    
    images = (image, imageBF1, imageBF2, imageDF)
    z = t2*c0/2

    xd2 = np.arange(numTxBeams)*transPitch
    xd2 = xd2 - np.max(xd2)/2

    # post process all images generated
    imagesProc = []
    for r in range(len(images)):
        
        im = images[r]
        
        # define portion of image you want to display
        # this includes nullifying beginning of image that contains the transmission pulse
    
        f = np.where(z < 5e-3)[0]
        zTrunc = np.delete(z,f)
        imTrunc = im[:,f[-1]+1:]

        # envelope detection
        for n in range(numTxBeams):
            imTrunc[n,:] = envDet(imTrunc[n,:], 2*zTrunc/c0 , method = 'hilbert')         #and add contributions across all 32 channels
            
        # log compression and scan conversion
        DR = 30  # dynamic range - units of dB
        reject = 0 # rejection level - units of dB 
        BG = 0 # brightness gain - units of dB
        imageLog = logCompress(imTrunc, DR, reject, BG)
            
        imageSC, zSC, xSC  = scanConv(imageLog, xd2, zTrunc) #convert to 512x512 image
    
        imageSC2 = np.round(255*imageSC/DR) #convert to 8-bit grayscale
        imageSC3 = imageSC2.astype('int')

        imagesProc.append(np.transpose(imageSC3))
        
    # plotting

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))

    ax1.imshow(imagesProc[0], extent=[xSC[0]*1e3,xSC[-1]*1e3,zSC[-1]*1e3,zSC[0]*1e3], cmap='gray', interpolation='none')
    ax1.set_ylabel('Depth(mm)')
    ax1.set_xlabel('x(mm)')
    ax1.set_title('No beamforming')
    
    ax2.imshow(imagesProc[1], extent=[xSC[0]*1e3,xSC[-1]*1e3,zSC[-1]*1e3,zSC[0]*1e3], cmap='gray',interpolation='none')
    ax2.set_ylabel('Depth(mm)')
    ax2.set_xlabel('x(mm)')
    ax2.set_title('Fixed Receive Focus at 15 mm')

    ax3.imshow(imagesProc[2], extent=[xSC[0]*1e3,xSC[-1]*1e3,zSC[-1]*1e3,zSC[0]*1e3], cmap='gray', interpolation='none')
    ax3.set_ylabel('Depth(mm)')
    ax3.set_xlabel('x(mm)')
    ax3.set_title('Fixed Receive Focus at 35 mm')

    ax4.imshow(imagesProc[3], extent=[xSC[0]*1e3,xSC[-1]*1e3,zSC[-1]*1e3,zSC[0]*1e3], cmap='gray', interpolation='none')
    ax4.set_ylabel('Depth(mm)')
    ax4.set_xlabel('x(mm)')
    ax4.set_title('Dynamic Focusing')
    plt.show()

if __name__ == '__main__':
    main()




