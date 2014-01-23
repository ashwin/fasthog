FastHOG
=======

The original **FastHOG** source files can be obtained [here](http://www.robots.ox.ac.uk/~lav/Papers/prisacariu_reid_tr2310_09/prisacariu_reid_tr2310_09.html).
These source files do not compile under any recent version of CUDA on Ubuntu (or any Linux distribution).

These source files were fixed to compile with CUDA 5.5 on Ubuntu 12.04.

Steps to compile and use this version of FastHOG:

1. Install CUDA 5.5 or a recent version.
2. Install `libxinerama-dev` and `libfreeimage-dev`.
3. Build and install the 2.0 branch of FLTK. Instructions to do this can be found [here](http://choorucode.com/2014/01/22/how-to-build-and-install-fltk-2-0/).
4. `cd source/fastHOG` and build using `make`.
5. Run the sample FastHOG program using `bin/release/fastHOG`. (Note that it has to be run from this directory, else it fails.) A picture of pedestrians is displayed. Click anywhere on it to detect the people.
