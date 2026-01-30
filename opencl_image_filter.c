#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "load_shader_file.h"
#include "jpeg_decoder.h"

/* function declarations */
unsigned char *load_jpeg_rgba(const char *, int *, int *);
void save_ppm(const char *, unsigned char *, int, int);

// #define WIDTH  1920
// #define HEIGHT 1080
// #define PIXELS (WIDTH * HEIGHT)

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s input.jpg output.ppm\n", argv[0]);
        return -1;
    }

    int width, height;
    unsigned char *image =
        load_jpeg_rgba(argv[1], &width, &height);

    size_t pixels = width * height;
    
    cl_int err;
    cl_uint num_platforms = 0;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("No OpenCL platforms found\n");
        return -1;
    }
    printf("clGetPlatformIDs Number of platforms: %d\n", num_platforms);

    /* 1. Platform + device */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs failed: %d\n", err);
        return -1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetDeviceIDs failed: %d\n", err);
        return -1;
    }

    /* 2. Context + queue */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue   = clCreateCommandQueue(context, device, 0, &err);

    /* 3. Load kernel from file */
    char *kernel_src = load_file("devide_by_two.cl");
    if (!kernel_src) {
        printf("Failed to load kernel file\n");
        return -1;
    }

    // /* 4. Input image (RGBA) */
    // unsigned char *image = malloc(PIXELS * 4);
    // for (int i = 0; i < PIXELS * 4; i++)
    //     image[i] = 200;   /* dummy gray image */

    /* 5. OpenCL buffer */
    cl_mem imgBuf = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   pixels * 4,
                                   image,
                                   &err);

    /* 6. Build program */
    cl_program program =
        clCreateProgramWithSource(context, 1,
                                  (const char **)&kernel_src, NULL, &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        printf("Build error:\n%s\n", log);
        return -1;
    }

    /* 7. Kernel */
    size_t buffer_size = width * height * sizeof(cl_uchar4);
    size_t total_pixels = buffer_size / sizeof(cl_uchar4);
    size_t global = total_pixels;

    cl_kernel kernel = clCreateKernel(program, "devide_by_two", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &imgBuf);
    clSetKernelArg(kernel, 1, sizeof(int), &((int){width}));
    clSetKernelArg(kernel, 2, sizeof(int), &total_pixels);

    /* 8. Run kernel */
    
    clEnqueueNDRangeKernel(queue, kernel,
                           1, NULL,
                           &global, NULL,
                           0, NULL, NULL);

    clFinish(queue);

    /* 9. Read result */
    clEnqueueReadBuffer(queue, imgBuf, CL_TRUE,
                         0, pixels * 4,
                         image, 0, NULL, NULL);

    /* 10. Verify */
    // printf("Pixel[0]   R=%d\n", image[0]);               // left side
    // printf("Pixel[end] R=%d\n", image[(width-1)*4]);     // right side

    save_ppm(argv[2], image, width, height);

    /* 11. Cleanup */
    clReleaseMemObject(imgBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(image);
    free(kernel_src);

    return 0;
}

