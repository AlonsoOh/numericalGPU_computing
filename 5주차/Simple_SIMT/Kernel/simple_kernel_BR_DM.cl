//
//  simple_kernel_BR_DM.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

__kernel void CombineTwoArrays( __global float* A, __global float* B, __global float* C  ) {
    int num_groups = get_num_groups(0);
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    
    int i = local_id * num_groups + group_id;
    
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
        C[i] = M_PI_F / (sin(A[i]) * cos(B[i]) + cos(A[i]) * sin(B[i]));
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
