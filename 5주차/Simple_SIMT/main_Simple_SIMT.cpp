#pragma warning(disable : 4996)

//
//  main_Simple_SIMT.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

#include<stdio.h>
#include<stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <time.h>

// #include <CL/cl.h>
#include "Util/my_OpenCL_util_2_1.h"
#include "config_SImple_SIMT.h"

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char* string;
} OPENCL_C_PROG_SRC;

typedef struct _OCL_STUFFS {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    OPENCL_C_PROG_SRC prog_src;
    cl_program program;
    cl_mem buffer_A, buffer_B, buffer_C_GPU;
    cl_event event_for_timing;
} OCL_STUFFS;

typedef struct _DYNAMIC_ARRAYS {
    float* array_A, * array_B, * array_C_CPU, * array_C_GPU;
} DYNAMIC_ARRAYS;

#include "misc.h"

FILE* fp_stat;
char tmp_string[512];

void combine_two_arrays_host(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = 1.0f / (sin(A[i])*cos(B[i]) + cos(A[i])*sin(B[i]));
    }
}

void combine_two_arrays_host_divergence(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        switch (i % 8) {
        case 0:
            C[i] = 1.0f / (sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
            break;
        case 1:
            C[i] = 0.0f;
            for (int k = 0; k < 64; k++) {
                C[i] += 1.0f / (sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
            }
            break;
        case 2:
            C[i] = 0.0f;
            for (int k = 0; k < 16; k++) {
                C[i] += (sin(A[i]) * cos(B[i]) + M_PI * cos(A[i]) * sin(B[i]));
            }
            break;
        case 3:
            C[i] = -1.0f / (2.0f * sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
            break;
        case 4:
            C[i] = M_PI / (sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
            break;
        case 5:
            C[i] = 0.0f;
            for (int k = 0; k < 32; k++) {
                C[i] += 1.0f / (M_PI * sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
            }
            break;
        case 6:
            C[i] = 0.0f;
            for (int k = 0; k < 64; k++) {
                C[i] -= (2.0f * sin(A[i]) * cos(B[i]) + M_PI * cos(A[i]) * sin(B[i]));
            }
            break;
        case 7:
            C[i] = -1.0f / (M_PI * sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
            break;
        }
    }
}
 
int combine_two_arrays_GPU_device(OCL_STUFFS* ocl_stuffs, DYNAMIC_ARRAYS* dynamic_arrays) {
    __int64 _start, _freq, _end;
    float compute_time;

    cl_int errcode_ret;

    /* Define the index space and work-group size */
    size_t global_work_size[1] = { GLOBAL_WORK_SIZE };
    size_t local_work_size[1] = { LOCAL_WORK_SIZE };

    /* Get the first platform. */
    errcode_ret = clGetPlatformIDs(1, &ocl_stuffs->platform_id, NULL);
    // You may skip error checking if you think it is unnecessary.
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Get the first GPU device. */
    errcode_ret = clGetDeviceIDs(ocl_stuffs->platform_id, CL_DEVICE_TYPE_GPU, 1, &ocl_stuffs->device_id, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
    print_device_0(ocl_stuffs->device_id);

    /* Create a context with the devices. */
     ocl_stuffs->context = clCreateContext(NULL, 1, &ocl_stuffs->device_id, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a command-queue for the GPU device. */
    // Use clCreateCommandQueueWithProperties() for OpenCL 2.0.
    ocl_stuffs->cmd_queue = clCreateCommandQueue(ocl_stuffs->context, ocl_stuffs->device_id, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a program from OpenCL C source code. */
    ocl_stuffs->prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME, &ocl_stuffs->prog_src.string);
        ocl_stuffs->program = clCreateProgramWithSource(ocl_stuffs->context, 1, (const char**)&ocl_stuffs->prog_src.string, 
        &ocl_stuffs->prog_src.length, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Build a program executable from the program object. */
    const char options[] = "-cl-std=CL1.2";
    errcode_ret = clBuildProgram(ocl_stuffs->program, 1, &ocl_stuffs->device_id, options, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(ocl_stuffs->program, ocl_stuffs->device_id, "GPU");
        return 1;
    }

    /* Create the kernel from the program. */
        ocl_stuffs->kernel = clCreateKernel(ocl_stuffs->program, KERNEL_NAME, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create input and output buffer objects. */
      ocl_stuffs->buffer_A = clCreateBuffer(ocl_stuffs->context, CL_MEM_READ_ONLY, 
        sizeof(float) * global_work_size[0], NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    ocl_stuffs->buffer_B = clCreateBuffer(ocl_stuffs->context, CL_MEM_READ_ONLY,
        sizeof(float) * global_work_size[0], NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    ocl_stuffs->buffer_C_GPU = clCreateBuffer(ocl_stuffs->context, CL_MEM_WRITE_ONLY, 
        sizeof(float) * global_work_size[0], NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "    [Data Transfer to GPU] \n");

    CHECK_TIME_START(_start, _freq);
    // Move the input data from the host memory to the GPU device memory.
    errcode_ret = clEnqueueWriteBuffer(ocl_stuffs->cmd_queue, ocl_stuffs->buffer_A, CL_FALSE, 0,
        sizeof(float) * global_work_size[0], dynamic_arrays->array_A, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    errcode_ret = clEnqueueWriteBuffer(ocl_stuffs->cmd_queue, ocl_stuffs->buffer_B, CL_FALSE, 0,
        sizeof(float) * global_work_size[0], dynamic_arrays->array_B, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Wait until all data transfers finish. */
    clFinish(ocl_stuffs->cmd_queue); // What if this line is removed?
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);

    /* Set the kenel arguments. */
    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 0, sizeof(cl_mem), &ocl_stuffs->buffer_A);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 1, sizeof(cl_mem), &ocl_stuffs->buffer_B);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 2, sizeof(cl_mem), &ocl_stuffs->buffer_C_GPU);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;


    printf_KernelWorkGroupInfo(ocl_stuffs->kernel, ocl_stuffs->device_id);

    fprintf(stdout, "    [Kernel Execution] \n");

    fp_stat = util_open_stat_file_append(STAT_FILE_NAME);
    util_stamp_stat_file_device_name_and_time(fp_stat, ocl_stuffs->device_id);
    util_reset_event_time();
    /* Execute the kernel on the device. */
     CHECK_TIME_START(_start, _freq);
     for (int i = 0; i < N_EXECUTIONS; i++) {
          errcode_ret = clEnqueueNDRangeKernel(ocl_stuffs->cmd_queue, ocl_stuffs->kernel, 1, NULL,
              global_work_size, local_work_size, 0, NULL, &ocl_stuffs->event_for_timing);
          clWaitForEvents(1, &ocl_stuffs->event_for_timing);
          // clFinish(ocl_stuffs->cmd_queue);
          if (CHECK_ERROR_CODE(errcode_ret)) return 1;
          util_accumulate_event_times_1_2(ocl_stuffs->event_for_timing);
     }
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    // print_device_time(ocl_stuffs->event_for_timing);
    util_print_accumulated_device_time_1_2(N_EXECUTIONS);
    // util_stamp_stat_file_ave_device_time_START_to_END_1_2(fp_stat, KERNEL_SELECTION, N_EXECUTIONS);
    MAKE_STAT_ITEM_LIST(tmp_string, global_work_size[0], local_work_size[0]);
    util_stamp_stat_file_ave_device_time_START_to_END_1_2_string(fp_stat, tmp_string);
    util_close_stat_file_append(fp_stat);

    fprintf(stdout, "    [Data Transfer] \n");

    /* Read back the device buffer to the host array. */
        CHECK_TIME_START(_start, _freq);
    errcode_ret = clEnqueueReadBuffer(ocl_stuffs->cmd_queue, ocl_stuffs->buffer_C_GPU, CL_TRUE, 0,
        sizeof(float) * global_work_size[0], dynamic_arrays->array_C_GPU, 0, NULL, &ocl_stuffs->event_for_timing);
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    // In this case, you do not need to call clFinish() for a synchronization.

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(ocl_stuffs->event_for_timing);

}

void compair_two_float_arrays(float* A, float* B, int n_elements, const char* array_type) {
    int n_differs = 0;

    for (int i = 0; i < n_elements; i++) {
        if ((double)fabs(A[i] - B[i])/fabs(A[i]) > 0.00001) { // Assume A[i] is not zero.
            n_differs++;
            fprintf(stdout, "   %f(A[%d]) != %f(B[%d])\n", A[i], i, B[i], i);
        }
    }
    if (n_differs == 0)
        fprintf(stdout, "      * The two %s are the same.\n", array_type);
    else
        fprintf(stdout, "      * There are %d discrepancies between the two %s.\n", n_differs, array_type);
}
 
int main(void) {
    if (0) {
        // Just to reveal my OpenCl platform...
        show_OpenCL_platform();
        return 0;
    }
 
    __int64 _start, _freq, _end;
    float compute_time;
    int n_elements = GLOBAL_WORK_SIZE;

    DYNAMIC_ARRAYS dynamic_arrays;
    dynamic_arrays.array_A = (float*)malloc(sizeof(float) * n_elements);
    dynamic_arrays.array_B = (float*)malloc(sizeof(float) * n_elements);
    dynamic_arrays.array_C_CPU = (float*)malloc(sizeof(float) * n_elements);
    dynamic_arrays.array_C_GPU = (float*)malloc(sizeof(float) * n_elements);

    if (!dynamic_arrays.array_A || !dynamic_arrays.array_B || !dynamic_arrays.array_C_CPU || !dynamic_arrays.array_C_GPU) {
        fprintf(stderr, "*** Error: cannot allocate memory for input/output arrays....\n");
        exit(-1);
    }

    /* STEP 0: Generate the two random input arrays. */
    fprintf(stdout, "\n^^^ STEP 0: Generate the two random input arrays with %d elements each...\n", n_elements);
    generate_random_float_array(dynamic_arrays.array_A, n_elements);
    generate_random_float_array(dynamic_arrays.array_B, n_elements);
    fprintf(stdout, "            Done!\n");

    /* STEP 1: combine the two arrays on the host. */
    fprintf(stdout, "\n^^^ STEP 1: Combine the two arrays on the host. ^^^\n");
    fprintf(stdout, "\n    [CPU Execution] \n");
    CHECK_TIME_START(_start, _freq);
#if (KERNEL_SELECTION == 0) || (KERNEL_SELECTION == 1)
    combine_two_arrays_host(dynamic_arrays.array_A, dynamic_arrays.array_B, dynamic_arrays.array_C_CPU, n_elements);
#else
    combine_two_arrays_host_divergence(dynamic_arrays.array_A, dynamic_arrays.array_B, dynamic_arrays.array_C_CPU, n_elements);
#endif

    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);

    MAKE_STAT_ITEM_LIST_CPU(tmp_string, compute_time);
    fp_stat = util_open_stat_file_append(STAT_FILE_NAME);
    util_stamp_stat_file_string_and_time(fp_stat, tmp_string);
    util_close_stat_file_append(fp_stat);

    fprintf(stdout, "    [Show Partial Results] \n");
    fprintf(stdout, "      * C[%d] = %f, C[%d] = %f, C[%d] = %f\n\n",
        n_elements / 3, dynamic_arrays.array_C_CPU[n_elements / 3],
        n_elements / 2, dynamic_arrays.array_C_CPU[n_elements / 2],
        3 * n_elements / 4, dynamic_arrays.array_C_CPU[3 * n_elements / 4]);

    /* STEP 2: Compute the two arrays on the GPU. */
    fprintf(stdout, "\n^^^ STEP 2: Compute the two arrays on the GPU. ^^^\n");

    OCL_STUFFS ocl_stuffs;

    int flag = combine_two_arrays_GPU_device(&ocl_stuffs, &dynamic_arrays);
    clean_up_ocl_stuffs(&ocl_stuffs);

    if (!flag) {
        /* STEP 3: Compare if the two output arrays from CPU and GPU are the same. */
        fprintf(stdout, "\n^^^ STEP 3: Compare if the two output arrays from CPU and GPU are the same. ^^^\n");
        fprintf(stdout, "\n    [Check Results]\n");
        compair_two_float_arrays(dynamic_arrays.array_C_CPU, dynamic_arrays.array_C_GPU, n_elements, "combined arrays");

        fprintf(stdout, "\n    [Show Partial Results] \n");
        fprintf(stdout, "      * C[%d] = %f, C[%d] = %f, C[%d] = %f\n\n",
            n_elements / 3, dynamic_arrays.array_C_GPU[n_elements / 3],
            n_elements / 2, dynamic_arrays.array_C_GPU[n_elements / 2],
            3 * n_elements / 4, dynamic_arrays.array_C_GPU[3 * n_elements / 4]);

        clean_up_dynamic_arrays(&dynamic_arrays);
        return 0;
    }
    else {
        clean_up_dynamic_arrays(&dynamic_arrays);
        return 1;
    }
    return 0;
}



