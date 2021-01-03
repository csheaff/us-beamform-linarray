# Intro
This is an example of ultrasound beamforming using a linear array in both Python and Rust. The Python script is written with something of an education emphasis, and the Rust script is a functional work-in-progress to be refined as Rust's signal and image processing libraries mature.

# Data Description
RF Data was simulated using a 3rd party MATLAB toolbox called K-Wave. Specifically, the data was generated using the program example_us_bmode_linear_transducer.m, which sets up a linear probe and generates the signals received after pulsing into a 3D scattering phantom. The phantom accounts for nonlinearity, multiple scattering, power law acoustic absorption, and a finite beam width in the elevation direction. One can find the nature of the simulated data as well as a description of the K-wave program [here][http://www.k-wave.org/documentation/example_us_bmode_linear_transducer.php].

The aforementioned m-file not only simulates recorded data but performs image reconstruction as well. However, I have merely acquired the raw RF data stored in the variable `sensor_data` and written my own image reconstruction routines.

# Python

The conventional steps in an ultrasound signal processing pipeline are conducted, and a comparison is performed between simple B-mode imaging, beamforming with fixed receive focus, and dynamic focusing:

![alt text](./py/result.png)

This program requires `numpy`, `scipy`, `matplotlib`, and `h5py`.

# Rust

The available signal and image processing libraries written in pure Rust are in their infancy at the time of this writing, so this script makes use of existing C++ libraries such as `fftw` and `opencv`. Currently a beamformed image is produced albeit without signal filtering and a slow upsampling step. The script is centered around `ndarray` and incorporates logging and plot creation/saving (`plotlib`) for the checking of intermediate outputs.






