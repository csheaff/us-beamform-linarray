extern crate hdf5;

extern crate basic_dsp;
use basic_dsp::conv_types::*;
use basic_dsp::*;

use ndarray::{prelude::*, stack, Zip};
use ndarray_linalg::{norm::Norm, types::Scalar};
use ndarray_stats::QuantileExt; // this adds basic stat methods to your arrays
//use ndarray_stats::SummaryStatisticsExt;
//use fftw::array::AlignedVec;
//use fftw::plan::*;
//use fftw::types::*;
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

    let record_length_2 = REC_LEN * INTERP_FACT;
    let mut data_apod = Array3::<f64>::zeros((
	N_PROBE_CHANNELS as usize, 
	N_TRANSMIT_BEAMS as usize,
	record_length_2 as usize,
    ));
    let mut buffer = SingleBuffer::new();
    for n in 0..N_PROBE_CHANNELS {
	for m in 0..N_TRANSMIT_BEAMS {
	    let mut waveform = data_apod.slice_mut(s![n as usize, m as usize, ..]);
	    // let mut dsp_vec = DspVec::<_, _, meta::Real, meta::Time>::from(waveform.to_owned().into_raw_vec());
	    let mut dsp_vec = waveform.to_owned().into_raw_vec().to_real_time_vec();
	    let waveform_interp = dsp_vec.interpolatei(&mut buffer, &RaisedCosineFunction::new(0.35), INTERP_FACT).unwrap();
	    // Problem here - I'm getting empty results.

	    // println!("{:?}", waveform_interp);
	    // waveform.assign(&waveform_interp.unwrap());
	}
    }

    
    let preproc_data = data.clone();
    let t_shifted = t.clone();
    (preproc_data, t_shifted)
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


fn beamform_df(preproc_data: &Array3<f64>, time: &Array1<f64>, xd: &Array1<f64>) -> Array2<f64> {

    // acoustic propagation distance from transmission to reception for each
    // element. Note: transmission is consdiered to arise from the center
    // of the array.
    let zd = time * SPEED_SOUND / 2.0;
    let zd2 = zd.mapv(|x| x.powi(2));
    let mut prop_dist = Array2::<f64>::zeros((N_PROBE_CHANNELS as usize, zd.len()));

    println!("{:?}", prop_dist);

    for r in 0..N_PROBE_CHANNELS {
	let dist = (xd[r as usize].powi(2) + &zd2).mapv(<f64>::sqrt);
	let mut slice = prop_dist.slice_mut(s![r as usize, ..]);
	slice.assign(&dist);
    }
    let mut prop_dist_ind = (prop_dist / SPEED_SOUND * SAMPLE_RATE).mapv(|x| x.round() as usize);
    
    // GET INTERPOLATION WORKING BEFORE PROCEEDING HERE

    // replace with last index (likely to be of low signal at that location i.e
    // closest to a null.
    // let is_outbounds = prop_dist_ind.mapv(|x| x >= time.len());
    // let outbounds_inds = where_2D(is_outbounds);
    
    // let replacement_ind = (time.len() - 1) as usize;
    // for oi in outbounds_inds.iter() {
	// let mut slice = prop_dist_ind.slice_mut(s![oi.0, oi.1]);
	// slice.assign(&replacement_ind);  // Clay, this is currently broken
    // }
        
    let mut image = Array2::<f64>::zeros((N_PROBE_CHANNELS as usize, zd.len()));
    
    return image
}


fn main() {
    let data_path = "../example_us_bmode_sensor_data.h5";
    let data = get_data(data_path);
    let t = Array::range(0.0, REC_LEN as f64, 1.0) / SAMPLE_RATE - TIME_OFFSET;
    let xd = Array::range(0.0, N_PROBE_CHANNELS as f64, 1.0) * ARRAY_PITCH;
    let xd_max = *xd.max().unwrap();
    let xd = xd - xd_max / 2.0;

    // println!("{:?}", xd_max);
    // println!("{:?}", 333333);

    let (preproc_data, t_shifted) = preproc(&data, &t, &xd);

    // let image = beamform_df(&preproc_data, &t_shifted, &xd);

}


// fn main() {
//     let bools = array![[false, true, false], [true, false, true]];
//     // let nonzero: Vec<_> = bools
//     //     .indexed_iter()
//     //     .filter_map(|(index, &item)| if item { Some(index) } else { None })
//     //     .collect();
//     let bools2 = where_2D(bools);
//     assert_eq!(bools2, vec![(0, 1), (1, 0), (1, 2)]);   
// }
