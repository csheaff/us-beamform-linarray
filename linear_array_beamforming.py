import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp2d
import h5py


# constants
num_tx_beams = 96
num_probe_chan = 32
tx_freq = 1.5e6
tx_focus = 20e-3
c0 = 1540
trans_pitch = 2*1.8519e-4
sample_rate = 27.72e6
toffset = 1.33e-6  # time at which the middle of the transmission pulse occurs


def arange2(start, stop=None, step=1):
    """#Modified version of numpy.arange which corrects error associated with non
    -integer step size"""
    if stop is None:
        a = np.arange(start)
    else: 
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a


def get_tGC(alpha0, prop_dist):
    """ Time-gain compensation
    The attenuation coefficient of tissue is usually expressed in d_b and
    defined as alpha_dB = 20/x*log10[p(0)/p(x)] where x is the propagation 
    distance, p(0) is the incident pressure, p(x) is the spatially variant 
    pressure.  As a result, p(x)/p(0) = 10^(-alphad_b*x/20). Additionally, 
    alpha is modeled as alpha_dB = alpha0*f^n where f is frequency and 0 < n < 1
    . For tissue, n ~ 1. alpha0 is usually the parameter specified in units of 
    d_b/(MHz-cm). We can compensate therefore by multiplying each A-line by 
    10^(alpha0*f*prop_dist*100/20). Note that this does not take into account
    the dissipation of acoustic energy with distance due to non-plane wave 
    propagation.

    inputs:  alpha0 - attenutation coefficient in d_b/(MHz-cm)
             prop_dist - round-trip propagation distance of acoustic pulse in 
                        meters

    outputs: tgc_gain - gain vector for multiplication with A-line """

    n = 1  # approx. 1 for soft tissue
    alpha = alpha0*(tx_freq*1e-6)**n
    tgc_gain = 10**(alpha*prop_dist*100/20)

    return tgc_gain


def preproc_uS(data, t, xd):
    """Analog time-gain compensation is typically applied followed by an 
    anti-aliasing filter (low-pass) and then A/D conversion. The input data is 
    already digitized here, so no need for anti-alias filtering. Following A/D
    conversion, one would ideally begin beamforming, however the summing process
    in beamforming can produce very high values if low frequencies are included. 
    This can result in the generation of a dynamic range in the data that
    exceeds what's allowable by the number of bits, thereby yielding data loss.
    Therefore it's necessary to high-pass filter before beamforming. In addition,
    beamforming is more accurate with a higher sampling rate because the 
    calculated beamforming delays are more accurately achieved. Hence 
    interpolation is used to upsample the signal. Finally, apodization is 
    applied before the beamformer.
    
    This preprocessing function therefore consists of:
    1) time-gain compensation
    2) filtering
    3) interpolation
    4) apodization 
    
    In the filtering step I've appied a band-pass, as higher frequencies are 
    also problematic and are usually addressed after beamforming. 
    
    inputs: data - transmission number x receive channel x time index
            t - time vector [s]
            xd - dector position vector [m]
    
    outputs: data_apod - processed data
             t2 - new time vectors    
    """

    sample_rate = 1/(t[1] - t[0])
    samples_per_acq = data.shape[2]
    
    a0 = 0.4
    
    # get time-gain compensation vectors based on estimate for propagation
    # distance to each element
    zd = t*c0/2
    zd2 = zd**2
    dist1 = zd
    tgc = np.zeros((num_probe_chan, samples_per_acq))
    for r in range(num_probe_chan):
        dist2 = np.sqrt(xd[r]**2 + zd2)
        prop_dist = dist1 + dist2
        tgc[r, :] = get_tGC(a0, prop_dist)

    # apply tgc
    data_amp = np.zeros(data.shape)
    for m in range(num_tx_beams):
        data_amp[m, :, :] = data[m, :, :]*tgc
        
    # retrieve filter coefficients

    filt_ord = 201
    lc, hc = 0.5e6, 2.5e6
    lc = lc/(sample_rate/2)  # normalize to nyquist frequency
    hc = hc/(sample_rate/2)
    B = signal.firwin(filt_ord, [lc, hc], pass_zero=False)  # band-pass filter

    # specify interpolation factor
    interp_fact = 4
    sample_rate = sample_rate*interp_fact
    samples_per_acq2 = samples_per_acq*interp_fact

    # get apodization window
    apod_win = signal.tukey(num_probe_chan)  # np.ones(num_probe_chan)

    # process
    data_apod = np.zeros((num_tx_beams, num_probe_chan, samples_per_acq2))
    for m in range(num_tx_beams):
        for n in range(num_probe_chan):
            w = data_amp[m, n, :]
            data_filt = signal.lfilter(B, 1, w)
            data_interp = signal.resample_poly(data_filt, interp_fact, 1)
            data_apod[m, n, :] = apod_win[n]*data_interp

    # create new time vector based on interpolation and filter delay
    freqs, delay = signal.group_delay((B, 1))
    delay = int(delay[0])*interp_fact
    t2 = np.arange(samples_per_acq2)/sample_rate + t[0] - delay/sample_rate
    
    # remove signal before t = 0
    f = np.where(t2 < 0)[0]
    t2 = np.delete(t2, f)
    data_apod = data_apod[:, :, f[-1]+1:]

    return data_apod, t2


def beamform(data, t, xd, receive_focus):
    """This employs the classic delay-and-sum method of beamforming entailing a 
    single focus location defined by receive_focus [m]. 
    inputs: 
     data - RF data, dimenions (transmission number, receive channel, time index)
     t - time vector associated with RF waveforms, [t] = seconds
     xd - horizontal position vector of receive channels relative to center, 
          [xd] = meters
     receive_focus - depth of focus for beamforming, [receive_focus] = meters
    outputs: 
     image - beamformed data, dimensions of (scanline index, depth)
    """
    Rf = receive_focus
    fs = 1/(t[1]-t[0])
    delay_ind = np.zeros(num_probe_chan, dtype=int)
    for r in range(num_probe_chan):
        delay = Rf/c0*(np.sqrt((xd[r]/Rf)**2+1)-1)   # difference between propagation time for a central element
        delay_ind[r] = int(round(delay*fs))           # and prop time for a off-centered element
    max_delay = np.max(delay_ind)
    
    waveform_length = data.shape[2]
    image = np.zeros((num_tx_beams, waveform_length))  # initialize
    for q in range(num_tx_beams):
        scan_line = np.zeros(waveform_length + max_delay)  # initialize
        for r in range(num_probe_chan):
            delay_pad = np.zeros(delay_ind[r])
            fill_pad = np.zeros(len(scan_line)-waveform_length-delay_ind[r])
            waveform = data[q, r, :]
            scan_line = scan_line + np.concatenate((fill_pad, waveform, delay_pad))
        image[q, :] = scan_line[max_delay:]
    return image


def beamform_dF(data, t, xd):
    """Ideally we could focus at all depths in receive when beamforming. This is
    done in an FPGA by using time delays that are time-varying. To clarify, 
    suppose we use the above beamform function to focus at some depth z0. Why 
    not use the delay to achieve this focus merely for the value at that depth?
    For some depth z0+dz, we would then have a new delay and use it to generate 
    the pixel only at z0+dz. So an array of time-dependent delay values can be
    generated for each channel that would allow focusing at each depth. 

    In order to achieve dynamic focusing offline, digitally, one could find the 
    time-dependent delays and apply them, but this would require operating a
    loop over each time value. One could also use the above beamform function 
    for each focal point and only keep the value generated for that depth, but
    again this would computationally wasteful. An alternative is to fill the 
    a-line first with values corresponding to the propagation time from emmission 
    to pixel to receiver. One can then simply index the signal received by an 
    element at the estimtae for propagation time and add that to the pixel, 
    followed by summing contributions from other channels. Focusing at all depths
    is effectively acheived, and this is the method applied below.
    
    inputs: 
            data - RF data, dimenions of (transmission number, receive channel, time index)
            t - time vector associated with RF waveforms, [t] = seconds
            xd - horizontal position vector of receive channels relative to center, [xd] = meters
            
    outputs: 
            image - beamformed data, dimensions of (scanline index, depth)

"""
    sample_rate = 1/(t[2]-t[1])
    zd = t*c0/2  # note this can be defined arbitrarily to get a higher resolution. 
    zd2 = zd**2  
    prop_dist = np.zeros((num_probe_chan, len(zd)))
    for r in range(num_probe_chan):
        dist1 = zd
        dist2 = np.sqrt(xd[r]**2+zd2)
        prop_dist[r, :] = dist1 + dist2
    prop_dist_ind = np.round(prop_dist/c0*sample_rate)
    prop_dist_ind = prop_dist_ind.astype('int')  # acoustic propagation distance from transmission to reception for each element
                                         # these distances stay the same as we slide across the aperture of the full array

    f = np.where(prop_dist_ind >= len(t))  # eliminate indices that are out of bounds
    prop_dist_ind[f[0], f[1]] = len(t)-1  # replace out-of-bound indices with last index (likely to be of low signal 
                                       # at that location i.e closest to a null
    scan_line = np.zeros(len(zd))
    image = np.zeros((num_tx_beams, len(zd)))
    for q in range(num_tx_beams):  # index transmission
        for r in range(num_probe_chan):  # index receiver
            v = data[q, r, :]      # get recorded waveform
            scan_line = scan_line + v[prop_dist_ind[r, :]]  # index waveform at times corresponding to propagation distance to pixel along a-line
        image[q, :] = scan_line
        scan_line = np.zeros(len(zd))
    return image


def env_det(scan_line, t, method='hilbert'):
    """Envelope detection. This can be done in a few ways: 
    (1) Hilbert transform method
        - doesn't require knowledge of carrier frequency
        - simple - doesn't require filtering
        - cannot be implement with analog electronics
        - edge effects are undesirable
      
    (2) Demodulation + Low-pass filtering
        - implementable with analog electronics
        - requires knowledge of the carrier frequency, which gets smaller with 
          propagation
        - more computational steps involved.

    'demod' and 'demod2' do exactly the same thing here. The former is merely 
    the simplest/most intuitive way to look at the operation (multiplying by 
    complex exponential yields a frequency shift in the fourier domain).
    Whereas with the latter, the I and Q components are defined, as is typical. 
    """
    n = 201
    fs = 1/(t[1]-t[0])
    lc = 0.75e6
    b = signal.firwin(n, lc/(fs/2))  # low-pass filter

    if method == 'hilbert':
        envelope = np.abs(signal.hilbert(scan_line))
    elif method == 'demod':
        demodulated = scan_line*np.exp(-1j*2*np.pi*tx_freq*t)
        demod_filt = np.sqrt(2)*signal.filtfilt(b, 1, demodulated)  # zero-phase filter to avoid time delay
        envelope = np.abs(demod_filt)
    elif method == 'demod2':
        I = scan_line*np.cos(2*np.pi*tx_freq*t)
        If = np.sqrt(2)*signal.filtfilt(b, 1, I)
        Q = scan_line*np.sin(2*np.pi*tx_freq*t)
        Qf = np.sqrt(2)*signal.filtfilt(b, 1, Q)
        envelope = np.sqrt(If**2+Qf**2)        
    return envelope


def log_compress(data, dynamic_range, reject_level, bright_gain):
    """Dynamic range is defined as the max value of some data divided by the 
    minimum value, and it is a measure of how spread out the data values are. 
    If the data values have been converted to d_b, then dynamic range is defined
    as the max value minus the minimum value. 

    One could interpret there being two stages of compression in the standard 
    log compression process. The first is the simple conversion to d_b. The second
    is in selecting to display a certain range of d_b. 

    inputs:
            data - envelope-detected data having values >= 0. Dimensions should 
                   be scanline x depth/time index
            dynamic_range - desired dynamic range of data to present [d_b]
            reject_level - level of rejection [d_b]
            bright_gain - brightness gain [d_b]
    output:
            xd_b3 - processed image, dimensions of scanline x depth/time index
    """

    # compress to dynamic range chosen
    xd_b = 20*np.log10(1+data)  # add 1 b/c log10(0) = -inf ('data' should have values >= 0)
    xd_b2 = xd_b - np.max(xd_b)  # shift such that max is 0 d_b
    xd_b3 = xd_b2 + dynamic_range  # shift such that max is dynamic_range value
    xd_b3[xd_b3 < 0] = 0  # eliminate data outside of dynamic range

    # rejection
    xd_b3[xd_b3 <= reject_level] = 0

    # add brightness gain, keep max value = dynamic_range
    xd_b3 = xd_b3 + bright_gain
    xd_b3[xd_b3 > dynamic_range] = dynamic_range
    
    return xd_b3


def scan_conv(data, xb, zb):
    """create 512x512 pixel image
    inputs: data - scanline x depth/time
            xb - horizontal distance vector
            zb - depth vector
    outputs: image_sC - scanline x depth/time
             znew - new depth vector
             xnew - new horizontal distance vector"""

    # decimate in depth dimensions
    decim_fact = 8
    data = data[:, 0:-1:decim_fact]
    zb = zb[0:-1:decim_fact]

    # interpolation
    interp_func = interp2d(zb, xb, data, kind='linear')
    dz = zb[1]-zb[0]
    xnew = arange2(xb[0], xb[-1]+dz, dz)  # make pixel square by making dx = dz
    znew = zb
    image_sC = interp_func(znew, xnew)

    return image_sC, znew, xnew


def main():

    h5f = h5py.File('example_us_bmode_sensor_data.h5', 'r')
    sensor_data = h5f['dataset_1'][:]

    # data get info
    samples_per_acq = sensor_data.shape[2]

    # time vector for data
    t = np.arange(samples_per_acq)/sample_rate - toffset
    
    # transducer locations relative to the a-line, which is always centered
    xd = np.arange(num_probe_chan)*trans_pitch
    xd = xd - np.max(xd)/2

    # preprocessing - signal filtering, interpolation, and apodization
    data_apod, t2 = preproc_uS(sensor_data, t, xd)

    # B-mode image w/o beamforming (only use waveform from central array element)
    image = data_apod[:, 15, :]
        
    # beamforming with different receive focii
    rx_focus = 15e-3
    image_bF1 = beamform(data_apod, t2, xd, rx_focus)
    
    rx_focus = 30e-3
    image_bF2 = beamform(data_apod, t2, xd, rx_focus)

    # beamforming with dynamic focusing
    image_dF = beamform_dF(data_apod, t2, xd)
    
    images = (image, image_bF1, image_bF2, image_dF)
    z = t2*c0/2

    xd2 = np.arange(num_tx_beams)*trans_pitch
    xd2 = xd2 - np.max(xd2)/2

    # post process all images generated
    images_proc = []
    for r in range(len(images)):
        
        im = images[r]
        
        # define portion of image you want to display
        # includes nullifying beginning of image containing transmission pulse
    
        f = np.where(z < 5e-3)[0]
        z_trunc = np.delete(z, f)
        im_trunc = im[:, f[-1]+1:]

        # envelope detection
        for n in range(num_tx_beams):
            im_trunc[n, :] = env_det(im_trunc[n, :], 2*z_trunc/c0,
                                   method='hilbert')
            
        # log compression and scan conversion
        DR = 35  # dynamic range - units of d_b
        reject = 0  # rejection level - units of d_b
        BG = 0  # brightness gain - units of d_b
        image_log = log_compress(im_trunc, DR, reject, BG)

        # convert to 512x512 image
        image_sC, z_sC, x_sC = scan_conv(image_log, xd2, z_trunc)
    
        image_sC2 = np.round(255*image_sC/DR)  # convert to 8-bit grayscale
        image_sC3 = image_sC2.astype('int')

        images_proc.append(np.transpose(image_sC3))
        
    # plotting

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    ax1.imshow(images_proc[0], extent=[x_sC[0]*1e3, x_sC[-1]*1e3, z_sC[-1]*1e3,
                                      z_sC[0]*1e3], cmap='gray', interpolation='none')
    ax1.set_ylabel('Depth(mm)')
    ax1.set_xlabel('x(mm)')
    ax1.set_title('No beamforming')
    
    ax2.imshow(images_proc[1], extent=[x_sC[0]*1e3, x_sC[-1]*1e3, z_sC[-1]*1e3,
                                      z_sC[0]*1e3], cmap='gray', interpolation='none')
    ax2.set_ylabel('Depth(mm)')
    ax2.set_xlabel('x(mm)')
    ax2.set_title('Fixed Receive Focus at 15 mm')

    ax3.imshow(images_proc[2], extent=[x_sC[0]*1e3, x_sC[-1]*1e3, z_sC[-1]*1e3,
                                      z_sC[0]*1e3], cmap='gray', interpolation='none')
    ax3.set_ylabel('Depth(mm)')
    ax3.set_xlabel('x(mm)')
    ax3.set_title('Fixed Receive Focus at 30 mm')

    ax4.imshow(images_proc[3], extent=[x_sC[0]*1e3, x_sC[-1]*1e3, z_sC[-1]*1e3,
                                      z_sC[0]*1e3], cmap='gray', interpolation='none')
    ax4.set_ylabel('Depth(mm)')
    ax4.set_xlabel('x(mm)')
    ax4.set_title('Dynamic Focusing')
        
    plt.show()


if __name__ == '__main__':
    main()
