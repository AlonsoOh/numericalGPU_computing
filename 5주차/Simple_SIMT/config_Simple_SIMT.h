//
//  config_Simple_SIMT.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

//		0 : NO_DIVERGENT_BRANCH & COALESCED_GLOBAL_MEMORY_ACCESS
//		1 : NO_DIVERGENT_BRANCH & DIVERGENT_GLOBAL_MEMORY_ACCESS
//		2 : DIVERGENT_BRANCH & COALESCED_GLOBAL_MEMORY_ACCESS
//		3 : DIVERGENT_BRANCH & DIVERGENT_GLOBAL_MEMORY_ACCESS

#define KERNEL_SELECTION  3

#if   KERNEL_SELECTION == 0
#define OPENCL_C_PROG_FILE_NAME "Kernel/simple_kernel_NB_CM.cl"
#elif KERNEL_SELECTION == 1
#define OPENCL_C_PROG_FILE_NAME "Kernel/simple_kernel_NB_DM.cl" 
#elif KERNEL_SELECTION == 2
#define OPENCL_C_PROG_FILE_NAME "Kernel/simple_kernel_BR_CM.cl" 
#elif KERNEL_SELECTION == 3
#define OPENCL_C_PROG_FILE_NAME "Kernel/simple_kernel_BR_DM.cl" 
#endif

#define KERNEL_NAME "CombineTwoArrays"

// size_t global_work_size[1] = { GLOBAL_WORK_SIZE };
#define GLOBAL_WORK_SIZE    (16 * 1024 * 1024) 
// size_t local_work_size[1] = { LOCAL_WORK_SIZE };
#define LOCAL_WORK_SIZE      (32)       

#define N_EXECUTIONS    5
#define STAT_FILE_NAME	"Data/stat_file_Simple_SIMT.txt"

#define MAKE_STAT_ITEM_LIST_CPU(string, time)  sprintf((string), "\n*** Host(CPU):: KER = %d, Time: %.3fms", \
			KERNEL_SELECTION, time);
#define MAKE_STAT_ITEM_LIST(string, gws, lws) sprintf((string), "GWS = %d, LWS = %d, KER = %d, N_EXE = %d",\
			(gws), (lws), KERNEL_SELECTION, N_EXECUTIONS);

