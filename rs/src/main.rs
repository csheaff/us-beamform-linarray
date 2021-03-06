#[macro_use] extern crate log;
extern crate simplelog;
use simplelog::*;
use std::fs::File;

use opencv::{
    core::{self, MatConstIterator, Point, Point2d, Rect, Size, Vec2d, Vec2b, Vec3d, Vec3f, Vec4w},
    Error,
    prelude::*,
    Result,
    types::{VectorOff64, VectorOfi32, VectorOfMat},
    imgproc::{self, resize},
};

extern crate hdf5;
extern crate basic_dsp;
use basic_dsp::conv_types::*;
use basic_dsp::*;

use ndarray::{prelude::*, stack, Zip};
use ndarray_linalg::{norm::Norm, types::Scalar};
use ndarray_stats::QuantileExt; // this adds basic stat methods to your arrays
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;

use std::f64::consts::PI;
use std::vec::Vec;
use std::path::Path;

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::LineStyle;

use std::time::Instant;

const SAMPLE_RATE: f64 = 27.72e6;
const TIME_OFFSET: f64 = 1.33e-6;
const SPEED_SOUND: f64 = 1540.0;
const N_TRANSMIT_BEAMS: u32 = 96;
const N_PROBE_CHANNELS: u32 = 32;
const TRANSMIT_FREQ: f64 = 1.6e6;
const TRANSMIT_FOCAL_DEPTH: f64 = 20e-3;
const ARRAY_PITCH: f64 = 2.0 * 1.8519e-4;
const REC_LEN: u32 = 1585;
const UPSAMP_FACT: u32 = 4;
const DECIM_FACT: u32 = 8;


fn plotlib_zip(x: &Array1<f64>, y: &Array1<f64>) -> Vec<(f64, f64)> {
    // recreate's python's zip() for two 1-d arrays, resulting in
    // a vector that can be digested by plotlib's Plot::new()
    let x = x.clone().into_raw_vec();
    let y = y.clone().into_raw_vec();
    let d: Vec<(f64, f64)> = x.into_iter().zip(y).collect();
    d
}


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


fn get_data(data_path: &Path) -> Array3<f64> {
    let file = hdf5::File::open(data_path).unwrap();
    let data = file.dataset("dataset_1").unwrap();
    let data: Array3<f64> = data.read().unwrap();
    return data
}


fn preproc(data: &Array3<f64>, t: &Array1<f64>, xd: &Array1<f64>) -> (Array3<f64>, Array1<f64>) {
    // Preprocessing. right now this only does upsampling/interpolation.
    // TODO: filtering, apodization, replace interpolation b/c
    //       it's slow
    let filt_ord = 201;
    let lc = 0.5e6;
    let uc = 2.5e6;
    let lc = lc / (SAMPLE_RATE / 2.0);
    let uc = uc / (SAMPLE_RATE / 2.0);

    let rec_len_interp = REC_LEN * UPSAMP_FACT;
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
	    dsp_vec.interpolatei(&mut buffer, &RaisedCosineFunction::new(0.1), UPSAMP_FACT).unwrap();
	    let (mut dsp_vec_data, points) = dsp_vec.get();
	    dsp_vec_data.truncate(points);
	    // let vec: Vec<f64> = dsp_vec.into(); // This also works but, what if you still need to operate on dsp_vec?

	    // plug into new array
	    let mut waveform_interp = data_interp.slice_mut(s![n as usize, m as usize, ..]);
	    waveform_interp.assign(&Array1::from(dsp_vec_data));
	}
    }
    let sample_rate = SAMPLE_RATE * UPSAMP_FACT as f64;
    let t_interp = Array::range(0.0, rec_len_interp as f64, 1.0) / sample_rate + t[0];

    // remove transmission pulse. truncating before 5 ms would be best,
    let trunc_ind = 350 as usize;
    let data_preproc = data_interp.slice(s![.., .., trunc_ind..]).into_owned();
    let t_interp = t_interp.slice(s![trunc_ind..]).into_owned();

    (data_preproc, t_interp)
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

    let sample_rate = SAMPLE_RATE * UPSAMP_FACT as f64;
    let prop_dist_ind = (prop_dist / SPEED_SOUND * sample_rate).mapv(|x| x.round() as usize);
    
    // replace out-of-bounds indices
    let prop_dist_ind = prop_dist_ind.mapv(|x| x.min(time.len() - 1));

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
    // Discrete-time analytic signal
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
    let nfft = 6340; // length of data
    let env = analytic(&waveform, nfft).mapv(|x| x.abs());
    let env = env.slice(s![..waveform.len()]).to_owned();

    env
}


fn log_compress(data: &Array2<f64>, dr: f64) -> Array2<f64> {
    let data_max = *(data.max().unwrap());
    let data_log = 20.0 * data.mapv(|x| (x / data_max).log10());
    let data_log = data_log.mapv(|x| x.max(-dr));
    let data_log = (data_log + dr) / dr;
    data_log
}


fn ndarray2mat_2d(x: &Array2<f64>) -> Mat {
    // covert a 2-d ndarray to single channel Mat object
    let n_rows = x.shape()[0];
    let x = x.clone().into_raw_vec();
    let mat = Mat::from_slice(&x).unwrap();
    let mat = mat.reshape(1, n_rows as i32).unwrap();
    mat
}


fn resize_ndarray(img_src: &Array2<f64>, fx: f64, fy: f64) -> Array2<f64> {
    // perform opencv's resize but using ndarray as input and output types
    // note: Array2::shape yields (height, width), but Mat::size yields (width, height)

    let n_rows_resize = (img_src.shape()[0] as f64 * fy) as i32;
    let n_cols_resize = (img_src.shape()[1] as f64 * fx) as i32;

    let img_src = ndarray2mat_2d(&img_src);
    let mut img_dst = Mat::default().unwrap();
    let size = Size::new(n_cols_resize, n_rows_resize);
    resize(&img_src, &mut img_dst, size, 0., 0., imgproc::INTER_LINEAR).unwrap();

    let img_dst = img_dst.data_typed::<f64>().unwrap().to_vec();
    let img_dst = Array2::<f64>::from_shape_vec((n_rows_resize as usize, n_cols_resize as usize), img_dst).unwrap();

    img_dst
}


fn scan_convert(img: &Array2<f64>, x: &Array1<f64>, z: &Array1<f64>)
		-> (Array2<f64>, Array1<f64>, Array1<f64>) {

    // decimate in depth dimensions
    let img_decim = img.slice(s![.., ..;DECIM_FACT]).into_owned();
    let z_sc = z.slice(s![..;DECIM_FACT]).into_owned();

    info!("Decimated imape shape = {:?}", img_decim.shape());

    // make pixels square by resampling in x-direction
    let dz = z_sc[1] - z_sc[0];
    let dx = x[1] - x[0];
    let img_sc = resize_ndarray(&img_decim, 1., dx / dz);
    let x_sc = Array1::<f64>::linspace(x[0], x[x.len() - 1], img_sc.shape()[0]);

    (img_sc, x_sc, z_sc)
}


fn transpose(a: Array2<f64>) -> Array2<f64> {
    // transpose a 2-d array while maintining c-order layout
    let a_t = a.t();
    let mut a_t_owned = Array2::zeros(a_t.raw_dim());
    a_t_owned.assign(&a_t);
    a_t_owned
}


fn img_save(img: &Array2<f64>, img_save_path: &Path) {
    // save grayscale image from Array2
    let img = img.clone();
    let img = 255.0 * img;
    let img = img.mapv(|x| x as u8);
    let imgx = img.shape()[0] as u32;
    let imgy = img.shape()[1] as u32;
    let imgbuf = image::GrayImage::from_vec(imgy, imgx, img.into_raw_vec());
    imgbuf.unwrap().save(img_save_path).unwrap();
}


fn logger_init(filename: &str) {
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Warn, Config::default(), TerminalMode::Mixed).unwrap(),
            WriteLogger::new(LevelFilter::Info, Config::default(), File::create(filename).unwrap()),
        ]
    ).unwrap();
}


fn main() {
    // logger and timer init
    let before = Instant::now();
    logger_init("rust.log");

    // data loading
    let data_path = Path::new("../example_us_bmode_sensor_data.h5");
    let data = get_data(&data_path);

    info!("Data shape = {:?}", data.shape());

    let t = Array::range(0., REC_LEN as f64, 1.) / SAMPLE_RATE - TIME_OFFSET;
    let xd = Array::range(0., N_PROBE_CHANNELS as f64, 1.) * ARRAY_PITCH;
    let xd_max = *xd.max().unwrap();
    let xd = xd - xd_max / 2.;

    // preprocessing
    let (preproc_data, t_interp) = preproc(&data, &t, &xd);
    let zd = &t_interp * SPEED_SOUND / 2.;

    info!("Preprocess Data shape = {:?}", preproc_data.shape());

    // beamforming
    let data_beamformed = beamform_df(&preproc_data, &t_interp, &xd);
    info!("Beamformed Data shape = {:?}", data_beamformed.shape());
    let m = data_beamformed.slice(s![0, ..]).sum();
    info!("Beamformed Data sum = {:?}", m);

    // lateral locations of beamformed a-lines
    let xd2 = Array1::<f64>::range(0., N_TRANSMIT_BEAMS as f64, 1.) * ARRAY_PITCH;
    let xd2_max = *xd.max().unwrap();
    let xd2 = xd2 - xd2_max / 2.;

    // envelope detection
    let mut img = Array2::<f64>::zeros(data_beamformed.raw_dim());
    for n in 0..N_TRANSMIT_BEAMS {
	let a_line = data_beamformed.slice(s![n as usize, ..]).into_owned();
	let env = envelope(&a_line);
	let mut img_slice = img.slice_mut(s![n as usize, ..]);
	img_slice.assign(&env);
    }
    info!("Envelope detected Data shape = {:?}", img.shape());

    // log compression
    let dr = 35.0;
    let img_log = log_compress(&img, dr);

    // scan conversion
    let (img_sc, x_sc, z_sc) = scan_convert(&img_log, &xd2, &zd);
    info!("Length of z vector after scan conversion {:?}", z_sc.len());
    info!("Length of x vector after scan conversion {:?}", x_sc.len());
    info!("Scan converted imape shape = {:?}", img_sc.shape());

    // image save
    let img_save_path = Path::new("./result.png");
    let img_sc = transpose(img_sc);
    img_save(&img_sc, &img_save_path);
    
    // optional plotting of intermediate steps
    let should_plot_steps = true;
    if should_plot_steps {

	// A-line vs pre-processed A-line
	let aline_ind = 15;
	let data_ex = data.slice(s![45, aline_ind, ..]).into_owned();
	let pp_ex = preproc_data.slice(s![45, aline_ind, ..]).into_owned();
	let xy = plotlib_zip(&t, &data_ex);
	let s1: Plot = Plot::new(xy).line_style(LineStyle::new()).legend(String::from("Waveform"));
	let xy = plotlib_zip(&t_interp, &pp_ex);
	let s2: Plot = Plot::new(xy).line_style(LineStyle::new().colour("#35C788")).legend(String::from("Preproc"));
	let v = ContinuousView::new().add(s1).add(s2).x_label("Time (s)").y_range(-5000., 5000.);
	Page::single(&v).save("./preproc.svg").unwrap();

	// Demo of Envelope detection
	let a_line = data_beamformed.slice(s![aline_ind, ..]).into_owned();
	let env = envelope(&a_line);
	let xy = plotlib_zip(&t_interp, &a_line);
	let s1: Plot = Plot::new(xy).line_style(LineStyle::new()).legend(String::from("Beamformed Waveform"));
	let xy = plotlib_zip(&t_interp, &env);
	let s2: Plot = Plot::new(xy).line_style(LineStyle::new().colour("#35C788")).legend(String::from("Envelope"));
	let v = ContinuousView::new().add(s1).add(s2).x_label("Time (s)");
	Page::single(&v).save("./envelope.svg").unwrap();

	//  of log compression
	let img_log_slice = img_log.slice(s![aline_ind, ..]).into_owned();
	let xy = plotlib_zip(&zd, &img_log_slice);
	let s1: Plot = Plot::new(xy).line_style(LineStyle::new());
	let v = ContinuousView::new().add(s1).x_label("Depth (m)");
	Page::single(&v).save("./img_log_slice.svg").unwrap();

    }

    info!("Elapsed time: {:.2?} s", before.elapsed());

    // TODO: filtering
    // TODO: replace basic_dsp interpolation (very slow)

}
