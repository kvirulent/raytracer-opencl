#include "main.h"

int main() {
    /* Setup & Configure OpenCL */
    // Identify availible platforms (compute devices)
    cl_platform_id platforms[64];
    unsigned int platform_count;
    cl_int platform_result = clGetPlatformIDs(64, platforms, &platform_count);

    // Get the first device on the first platform
    // maybe replace this with actually identifying a 
    // proper device and throwing an error if unable to...
    cl_device_id devices[1];
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, nullptr);
    cl_device_id device = devices[0];

    // Create run context and command queue 
    cl_context context = clCreateContext(nullptr, 1, & device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Fetch & build kernel source
    std::string kernel_src_str = fetch_src("vecsum");
    const char* kernel_src = kernel_src_str.c_str(); // convert to c-style
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, 0, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "vecsum", nullptr);

    /* Setup input & output buffers */
    // Create random data to give to vecsum
    float veca_data[256];
    float vecb_data[256];

    for (int i = 0; i < 256; ++i) {
        veca_data[i] = (float)(i * i);
        vecb_data[i] = (float)i;
    }

    // Create I/O buffers and enqueue writes
    cl_mem ibf_veca = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), nullptr, nullptr); // in buffer
    cl_mem ibf_vecb = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), nullptr, nullptr); // in buffer
    cl_mem obf_vecc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(float), nullptr, nullptr); // out buffer
    clEnqueueWriteBuffer(queue, ibf_veca, CL_TRUE, 0, 256 * sizeof(float), veca_data, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, ibf_vecb, CL_TRUE, 0, 256 * sizeof(float), vecb_data, 0, nullptr, nullptr);


    // Add buffers as kernel args so the kernel can access them
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &ibf_veca);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ibf_vecb);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &obf_vecc);

    // Use 256 cores in 64-core local blocks
    size_t global_work_size = 256;
    size_t local_work_size = 64;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_work_size, &local_work_size, 0, nullptr, nullptr);

    // Enqueue output buffer read
    float vecc_data[256];
    clEnqueueReadBuffer(queue, obf_vecc, CL_TRUE, 0, 256 * sizeof (float), vecc_data, 0, nullptr, nullptr);

    // Yield until queue is finished
    clFinish(queue);

    // Print vecsum output buffer
    std::cout << "Result:\n";
    for (int i = 0; i < 256; ++i) {
        std::cout << vecc_data[i] << std::endl;
    }

    // Release all objects
    clReleaseMemObject(ibf_veca);
    clReleaseMemObject(ibf_vecb);
    clReleaseMemObject(obf_vecc);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
}