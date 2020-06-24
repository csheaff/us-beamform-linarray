extern crate hdf5;
use ndarray::{prelude::*, stack, Zip};
use ndarray_linalg::{norm::Norm, types::Scalar};
use ndarray_stats::QuantileExt; // this adds basic stat methods to your arrays
//use ndarray_stats::SummaryStatisticsExt;
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use num_integer::Roots;
use std::f64::consts::PI;
use std::time::Instant;
use hdf5::File;

fn get_data(data_path: &str) -> Array3<f64> {
    let file = hdf5::File::open(data_path).unwrap();
    let data = file.dataset("dataset_1").unwrap();
    let data: Array3<f64> = data.read().unwrap();
    return data
}

fn main() {
    let data_path = "../example_us_bmode_sensor_data.h5";
    let data = get_data(data_path);
    println!("{:?}", data.shape())
}
