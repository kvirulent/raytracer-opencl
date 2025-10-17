# rt-opencl
A basic raytracer built in OpenCL and closely following ssloy's [tinyraytracer]() guide.

## Structure
- `/include` : External headers such as stb_image_write.h
- `/src` : Source code folder
    - `main.cpp/.h` : Main program entry point
    - `/kernels` : Contains OpenCL kernels
        - `/fetch.cpp/.h` : Utility for fetching kernel source code
        - `/src` : Contains OpenCL kernel source code
            - `gradient.cl` : Paints a mango gradient into the output buffer
            - `vecsum.cl` : Sums vectors of *n* size
            - `intersect_rs.cl` : Calculates the intersect of a ray and a sphere

## Build & Run
You will need to install a useable driver and the OpenCL SDK/headers. Refer to Khronos Groups' [OpenCL Guide]() where you can find information about how to install the SDK and driver for your platform and other fundamental information regarding OpenCL.

A full cmake configuration is included with the source code. You will need to configure the makefile first using cmake:
```
cmake -S . -B ./build
```
Then use cmake to build the project:
```
cmake --build build
```
This will export build artifacts (kernel source & executable) to `./out`. Running the executable will produce `out.png` within your working directory which is the output image from the raytracer.

## Work In Progress
It should be clear that this program is unfinished. Below is a tracker of implementation progress.

- [x] Ray-sphere intersect kernel
- [ ] Allow several sphere objects to be processed
- [ ] Scene descriptors like materials and lights
- [ ] Blinn-Phong shading
- [ ] Shadows
- [ ] Reflections / Refractions
- [ ] Use HDRIs for background images
- [ ] Render triangle meshes