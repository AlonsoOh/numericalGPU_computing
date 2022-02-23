//
//  misc.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

#pragma once

void clean_up_ocl_stuffs(OCL_STUFFS* ocl_stuffs) {
    // Free OpenCL resources. 
    if (ocl_stuffs->prog_src.string)
        free(ocl_stuffs->prog_src.string);

    if (ocl_stuffs->buffer_A)
        clReleaseMemObject(ocl_stuffs->buffer_A);
    if (ocl_stuffs->buffer_B)
        clReleaseMemObject(ocl_stuffs->buffer_B);
    if (ocl_stuffs->buffer_C_GPU)
        clReleaseMemObject(ocl_stuffs->buffer_C_GPU);

    if (ocl_stuffs->kernel)
        clReleaseKernel(ocl_stuffs->kernel);
    if (ocl_stuffs->program)
        clReleaseProgram(ocl_stuffs->program);
    if (ocl_stuffs->cmd_queue)
        clReleaseCommandQueue(ocl_stuffs->cmd_queue);
    if (ocl_stuffs->device_id)
        clReleaseDevice(ocl_stuffs->device_id);
    if (ocl_stuffs->context)
        clReleaseContext(ocl_stuffs->context);
}

void clean_up_dynamic_arrays(DYNAMIC_ARRAYS* dynamic_arrays) {
    // Free OpenCL resources. 
    if (dynamic_arrays->array_A)
        free(dynamic_arrays->array_A);
    if (dynamic_arrays->array_B)
        free(dynamic_arrays->array_B);
    if (dynamic_arrays->array_C_CPU)
        free(dynamic_arrays->array_C_CPU);
    if (dynamic_arrays->array_C_GPU)
        free(dynamic_arrays->array_C_GPU);
}

void generate_random_float_array(float* array, int n) {
    srand((unsigned int)201803); // Always the same input data
    for (int i = 0; i < n; i++) {
        array[i] = 3.1415926f * ((float)rand() / RAND_MAX);
    }
}
