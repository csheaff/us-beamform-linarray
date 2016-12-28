This is an example of ultrasound beamforming using a linear array in Python. Data was simulated using a 3rd party MATLAB toolbox called K-Wave created by Bradley Treeby, Ben Cox, and Jiri Jaros. Specifically, the data was generated using the program 'example_us_bmode_linear_transducer.m'. This program not only simulates recorded data but performs image reconstruction as well. However, I have merely acquired the raw RF data stored in the variable 'sensor_data' and written my own image reconstruction routines. A comparison is performed between simple B-mode imaging, beamforming with fixed recieve focus, and dynamic focusing.

One can find the nature of the simulated data as well as a description of the K-wave program here: 

http://www.k-wave.org/documentation/example_us_bmode_linear_transducer.php

Here are some attributes of the the simulated data, as mentioned in the comments for the K-wave program:
"Note, this example generates a B-mode ultrasound image from a 3D scattering phantom using kspaceFirstOrder3D. Compared to ray-tracing or Field II, this approach is very general. In particular, it accounts for nonlinearity, multiple scattering, power law acoustic absorption, and a finite beam width in the elevation direction. 

The RF data is stored in the file named "example_us_bmode_sensor_data.mat"