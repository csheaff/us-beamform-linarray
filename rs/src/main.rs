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


fn main() {
    println!("Hello, world!");
}
