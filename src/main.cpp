#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "main.h"

int main() {
    std::cout << "hello" << std::endl;
    /* Setup & Configure OpenCL */
    // Identify availible platforms (compute devices)
    cl_platform_id platforms[64];
    unsigned int platform_count;
    cl_int platform_result = clGetPlatformIDs(64, platforms, &platform_count);

    // Get the first device on the first platform
    cl_device_id device;
    cl_device_id devices[64];
    unsigned int device_qty;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 64, devices, &device_qty);
    for (int i = 0; i < device_qty; ++i) {
        char device_name[256];
        cl_int ok = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256, device_name, nullptr);
        if (ok == CL_SUCCESS && std::string(device_name) == TARGET_DEVICE_NAME) {
            std::cout << "Using device " << device_name << std::endl;
            device = devices[i];
            break;
        }
    }
    
    // Create run context and command queue 
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Fetch & build kernel source
    std::string kernel_src_str = fetch_src("gradient");
    const char* kernel_src = kernel_src_str.c_str(); // convert to c-style
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, 0, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "gradient", nullptr);
    std::cout << "Compiled kernel" << std::endl;

    const int WIDTH = 2048;
    const int HEIGHT = 1536;
    const size_t px = static_cast<size_t>(WIDTH) * HEIGHT;

    cl_mem ibf_dim_x = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, nullptr);
    cl_mem ibf_dim_y = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, nullptr);
    cl_mem obf_frame = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (WIDTH * HEIGHT) * sizeof(cl_float3), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, ibf_dim_x, CL_TRUE, 0, 1 * sizeof(int), &WIDTH, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, ibf_dim_y, CL_TRUE, 0, 1 * sizeof(int), &HEIGHT, 0, nullptr, nullptr);
    std::cout << "Enqueued buffer writes" << std::endl;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &ibf_dim_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ibf_dim_y);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &obf_frame);
    std::cout << "Added buffers to kernel args" << std::endl;

    size_t global_work_size[2] = { WIDTH, HEIGHT };
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    std::cout << "Enqueued kernel" << std::endl;

    std::vector<cl_float3> framebuf(px);
    clEnqueueReadBuffer(queue, obf_frame, CL_TRUE, 0, (WIDTH * HEIGHT) * sizeof (cl_float3), framebuf.data(), 0, nullptr, nullptr);
    std::cout << "Enqueued obf framebuf read" << std::endl;

    // Yield until queue is finished
    clFinish(queue);
    std::cout << "clFinish-ed" << std::endl;

    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < HEIGHT * WIDTH; i++) {
        cl_float3 &c = framebuf.at(i);
        Vec3f t(c.s[0], c.s[1], c.s[2]); // convert to vec3f for easy math
        float max = std::max(t[0], std::max(t[1], t[2]));
        for (size_t j = 0; j < 3; j++) {
            bytes.push_back((char)(255 * std::max(0.f, std::min(1.f, t[j]))));
        }
    }

    // Release all objects
    clReleaseMemObject(ibf_dim_x);
    clReleaseMemObject(ibf_dim_y);
    clReleaseMemObject(obf_frame);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);

    int r = stbi_write_png("./out.png", WIDTH, HEIGHT, 3, bytes.data(), WIDTH * 3);
    std::cout << "Image write returned " << r << std::endl;
    return r;
}