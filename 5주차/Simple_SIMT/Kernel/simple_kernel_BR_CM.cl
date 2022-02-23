//
//  simple_kernel_BR_CM.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

// By uncommenting the qualifier "__attribute__((reqd_work_group_size(128, 1, 1)))", you may specify
// the work-group size that must be used as the local_work_size argument to clEnqueueNDRangeKernel.
// This allows the compiler to optimize the generated code appropriately for this kernel.

__kernel
//__attribute__((reqd_work_group_size(128, 1, 1)))
void CombineTwoArrays(__global float* A, __global float* B, __global float* C) {
    int i = get_global_id(0);

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
            C[i] += (sin(A[i]) * cos(B[i]) + M_PI_F * cos(A[i]) * sin(B[i]));
        }
        break;
    case 3:
        C[i] = -1.0f / (2.0f * sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
        break;
    case 4:
        C[i] = M_PI_F/ (sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
        break;
    case 5:
        C[i] = 0.0f;
        for (int k = 0; k < 32; k++) {
            C[i] += 1.0f / (M_PI_F * sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
        }
        break;
    case 6:
        C[i] = 0.0f;
        for (int k = 0; k < 64; k++) {
            C[i] -= (2.0f * sin(A[i]) * cos(B[i]) + M_PI_F * cos(A[i]) * sin(B[i]));
        }
        break;
    case 7:
        C[i] = -1.0f / (M_PI_F * sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
        break;
    }
}
