extern crate hdf5;
extern crate basic_dsp;
use basic_dsp::conv_types::*;
use basic_dsp::*;

use ndarray::{prelude::*, stack, Zip};
use ndarray_linalg::{norm::Norm, types::Scalar};
use ndarray_stats::QuantileExt; // this adds basic stat methods to your arrays
//use ndarray_stats::SummaryStatisticsExt;
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
//use num_integer::Roots;
use std::f64::consts::PI;
//use std::time::Instant;
use std::vec::Vec;
use hdf5::File;

const SAMPLE_RATE: f64 = 27.72e6;
const TIME_OFFSET: f64 = 1.33e-6;
const SPEED_SOUND: f64 = 1540.0;
const N_TRANSMIT_BEAMS: u32 = 96;
const N_PROBE_CHANNELS: u32 = 32;
const TRANSMIT_FREQ: f64 = 1.6e6;
const TRANSMIT_FOCAL_DEPTH: f64 = 20e-3;
const ARRAY_PITCH: f64 = 2.0 * 1.8519e-4;
const REC_LEN: u32 = 1585;
const INTERP_FACT: u32 = 4;


fn fft_priv(x: &Array1<c64>, n: usize, sign: Sign) -> Array1<c64> {
    let mut xfft = AlignedVec::new(n);
    let mut xs_aligned = AlignedVec::new(n);
    for (x_aligned, &x) in xs_aligned.iter_mut().zip(x) {
        *x_aligned = x;
    }

    let mut plan: C2CPlan64 = C2CPlan::aligned(&[n], sign, Flag::MEASURE).unwrap();

    plan.c2c(&mut xs_aligned, &mut xfft).unwrap();
    Array1::from(Vec::from(xfft.as_slice()))
}

fn fft(x: &Array1<c64>, n: usize) -> Array1<c64> {
    // this is unnormalized, just like scipy.fftpack.fft

    fft_priv(x, n, Sign::Forward)
}

fn ifft(x: &Array1<c64>) -> Array1<c64> {
    // this will normalize, just like scipy.fftpack.ifft

    fft_priv(x, x.len(), Sign::Backward) / c64::new(x.len() as f64, 0.0)
}


fn get_data(data_path: &str) -> Array3<f64> {
    let file = hdf5::File::open(data_path).unwrap();
    let data = file.dataset("dataset_1").unwrap();
    let data: Array3<f64> = data.read().unwrap();
    return data
}


fn preproc(data: &Array3<f64>, t: &Array1<f64>, xd: &Array1<f64>) -> (Array3<f64>, Array1<f64>) {    

    let filt_ord = 201;
    let lc = 0.5e6;
    let uc = 2.5e6;
    let lc = lc / (SAMPLE_RATE / 2.0);
    let uc = uc / (SAMPLE_RATE / 2.0);

    // TODO: Set up filter coefs and apod window

    let rec_len_interp = REC_LEN * INTERP_FACT;
    let mut data_interp = Array3::<f64>::zeros((
	N_TRANSMIT_BEAMS as usize,
	N_PROBE_CHANNELS as usize,
	rec_len_interp as usize,
    ));
    let mut buffer = SingleBuffer::new();
    for n in 0..N_TRANSMIT_BEAMS {
	for m in 0..N_PROBE_CHANNELS {
	    // get waveform and convert to DspVec<f64>
	    let waveform = data.slice(s![n as usize, m as usize, ..]);
	    let mut dsp_vec = waveform.to_owned().into_raw_vec().to_real_time_vec();

	    // interpolate - currently a bug(ish) requiring truncation. See https://github.com/liebharc/basic_dsp/issues/46
	    dsp_vec.interpolatei(&mut buffer, &RaisedCosineFunction::new(0.35), INTERP_FACT).unwrap();
	    let (mut dsp_vec_data, points) = dsp_vec.get();
	    dsp_vec_data.truncate(points);	    
	    // let vec: Vec<f64> = dsp_vec.into(); // This also works but, what if you still need to operate on dsp_vec?

	    // plug into new array
	    let mut waveform_interp = data_interp.slice_mut(s![n as usize, m as usize, ..]);
	    waveform_interp.assign(&Array1::from(dsp_vec_data));
	}
    }
    let sample_rate = SAMPLE_RATE * INTERP_FACT as f64;
    let t_interp = Array::range(0.0, rec_len_interp as f64, 1.0) / sample_rate - TIME_OFFSET;
    (data_interp, t_interp)
}


fn where_2D(bools: Array2<bool>) -> Vec<(usize, usize)> {
    // This is designed to behave like np.where. Currently ndarray does not
    // provide this function natively. See https://github.com/rust-ndarray/ndarray/issues/466	
    let x: Vec<_> = bools
        .indexed_iter()
        .filter_map(|(index, &item)| if item { Some(index) } else { None })
	.collect();
    return x
}


fn array_indexing_1d(x: &Array1<f64>, ind: &Array1<usize>) -> Array1<f64> {
    Zip::from(ind).apply_collect(|idx| x[*idx])
}


fn beamform_df(data: &Array3<f64>, time: &Array1<f64>, xd: &Array1<f64>) -> Array2<f64> {
    // acoustic propagation distance from transmission to reception for each
    // element. Note: transmission is consdiered to arise from the center
    // of the array.
    let zd = time * SPEED_SOUND / 2.0;
    let zd2 = zd.mapv(|x| x.powi(2));
    let mut prop_dist = Array2::<f64>::zeros((N_PROBE_CHANNELS as usize, zd.len()));
    for r in 0..N_PROBE_CHANNELS {
	let dist = (xd[r as usize].powi(2) + &zd2).mapv(<f64>::sqrt) + &zd;
	let mut slice = prop_dist.slice_mut(s![r as usize, ..]);
	slice.assign(&dist);
    }
    let sample_rate = SAMPLE_RATE * INTERP_FACT as f64;
    let mut prop_dist_ind = (prop_dist / SPEED_SOUND * sample_rate).mapv(|x| x.round() as usize);

    // replace with last index (likely to be of low signal at that location i.e
    // closest to a null.
    // NOTE: oob_inds is currently empty because you haven't truncated
    //       the iterpolated time vector yet (it's not necessary until
    //       after filtering.   
    let is_oob = prop_dist_ind.mapv(|x| x >= time.len());
    let oob_inds = where_2D(is_oob);
    let replacement_ind: Array1<usize> = array![(time.len() - 1)];
    for oob_ind in oob_inds.iter() {
	let mut slice = prop_dist_ind.slice_mut(s![oob_ind.0, oob_ind.1]);
	slice.assign(&replacement_ind);
    }
    
    // beamform
    let mut image = Array2::<f64>::zeros((N_TRANSMIT_BEAMS as usize, zd.len()));
    for n in 0..N_TRANSMIT_BEAMS {
        let mut scan_line = Array1::<f64>::zeros(zd.len());
	for m in 0..N_PROBE_CHANNELS {
	    let waveform = data.slice(s![n as usize, m as usize, ..]).into_owned();
	    let inds = prop_dist_ind.slice(s![m as usize, ..]).into_owned();
	    let waveform_indexed = array_indexing_1d(&waveform, &inds);
	    scan_line += &waveform_indexed;
	}
	let mut image_slice = image.slice_mut(s![n as usize, ..]);
	image_slice.assign(&scan_line);
    }
    return image
}


fn analytic(waveform: &Array1<f64>, nfft: usize) -> Array1<c64> {
    //// Discrete-time analytic signal

    // This mimics scipy.signal.hilbert

    let waveform = waveform.mapv(|x| c64::new(x, 0.0)); // convert to complex
    let waveform_fft = fft(&waveform, nfft);

    // currently only working if nfft is even
    let mut h1 = Array1::<f64>::ones(nfft);
    let h2 = Array1::<f64>::ones(((nfft / 2) - 1) as usize) * 2.0;
    let mut slice = h1.slice_mut(s![1..(nfft / 2)]);
    slice.assign(&h2);
    let h0 = Array1::<f64>::zeros((nfft / 2 - 1) as usize);
    let mut slice = h1.slice_mut(s![(nfft / 2) + 1..]);
    slice.assign(&h0);
    
    let analytic_fft = waveform_fft * h1.mapv(|x| c64::new(x, 0.0));
    let analytic = ifft(&analytic_fft);

    analytic
}


fn envelope(waveform: &Array1<f64>) -> Array1<f64> {
    let nfft = 2048;
    let env = analytic(&waveform, nfft).mapv(|x| x.abs());
    let env = env.slice(s![..waveform.len()]).to_owned();

    env
}


 //TODO figure out how to implement Hilbert transform.
    //  Unfortunately their isn't an off-the-shelf function for computing
    // the magnitude of the analytic signal. However, the algorithm for this
    // appears somewhat simple. see:
    // https://www.mathworks.com/help/signal/ref/hilbert.html

    


// }


// fn log_compress() {


// }


// fn scan_convert() {


// }


fn main() {
    // let data_path = "../example_us_bmode_sensor_data.h5";
    // let data = get_data(data_path);
    // let t = Array::range(0.0, REC_LEN as f64, 1.0) / SAMPLE_RATE - TIME_OFFSET;
    // let xd = Array::range(0.0, N_PROBE_CHANNELS as f64, 1.0) * ARRAY_PITCH;
    // let xd_max = *xd.max().unwrap();
    // let xd = xd - xd_max / 2.0;

    // let (preproc_data, t_interp) = preproc(&data, &t, &xd);

    // let image = beamform_df(&preproc_data, &t_interp, &xd);

    let x = array![0., 1., 2., 3., 4., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 4., 3., 2., 1., 0.];
    let env = envelope(&x);
    println!("{:?}", env);

    // TODO
    // envelope detection
    // log compression
    // scan conversion


}
