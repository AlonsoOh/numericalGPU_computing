//
//  Config_CBO.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////////////
#define		STAT_FILE_NAME				"Data/stat_file_CBO.txt"
#define		INPUT_IMAGE					9
#define		N_EXECUTIONS				7

#define		LOCAL_WORK_SIZE_0			16		// Dim 0 (x)
#define		LOCAL_WORK_SIZE_1			8		// Dim 1 (y)

#define		GAUSSIAN_FILTER_SIZE		5		// 5, 7, 9, or 11
#define		KERNEL_SELECTION			25		//select part
/////////////////////////////////////////////////////////////////////////////////////
#if   KERNEL_SELECTION == 0
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_naive.cl"
#define KERNEL_NAME "convolution_naive"
#elif KERNEL_SELECTION == 1
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4.cl"
#define KERNEL_NAME "convolution_vec4"
#elif KERNEL_SELECTION == 11
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_mod.cl"
#define KERNEL_NAME "convolution_vec4_mod"
#elif KERNEL_SELECTION == 15	//  Must be GAUSSIAN_FILTER_SIZE = 5
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_LU.cl"
#define KERNEL_NAME "convolution_vec4_LU_5"
#elif KERNEL_SELECTION == 17	//  Must be GAUSSIAN_FILTER_SIZE	= 7
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_LU.cl"
#define KERNEL_NAME "convolution_vec4_LU_7"
#elif KERNEL_SELECTION == 19	//  Must be GAUSSIAN_FILTER_SIZE	= 9
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_LU.cl"
#define KERNEL_NAME "convolution_vec4_LU_9"
#elif KERNEL_SELECTION == 111  //  Must be GAUSSIAN_FILTER_SIZE	= 11
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_LU.cl"
#define KERNEL_NAME "convolution_vec4_LU_11"
#elif KERNEL_SELECTION == 2
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_local.cl"
#define KERNEL_NAME "convolution_vec4_local"
#elif KERNEL_SELECTION == 25
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_local_LU.cl"
#define KERNEL_NAME "convolution_vec4_local_LU_5" //  Must be GAUSSIAN_FILTER_SIZE = 5
#elif KERNEL_SELECTION == 27
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_local_LU.cl"
#define KERNEL_NAME "convolution_vec4_local_LU_7" //  Must be GAUSSIAN_FILTER_SIZE = 7
#elif KERNEL_SELECTION == 29
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_local_LU.cl"
#define KERNEL_NAME "convolution_vec4_local_LU_9" //  Must be GAUSSIAN_FILTER_SIZE = 9
#elif KERNEL_SELECTION == 211
#define OPENCL_C_PROG_FILE_NAME "Kernel/convolution_vec4_local_LU.cl"
#define KERNEL_NAME "convolution_vec4_local_LU_11" //  Must be GAUSSIAN_FILTER_SIZE = 11
#endif
/////////////////////////////////////////////////////////////////////////////////////
#define INPUT_FILE_0		"Image_0_7360_4832"
#define INPUT_FILE_1		"Image_1_9984_6400"
#define INPUT_FILE_2		"Image_2_7680_4320"
#define INPUT_FILE_3		"Image_3_8960_5408"
#define INPUT_FILE_4		"Image_4_6304_4192"
#define INPUT_FILE_5		"Image_5_1856_1376"
#define INPUT_FILE_8		"Grass_texture_2048_2048"
#define INPUT_FILE_9		"Tiger_texture_512_512"

#if INPUT_IMAGE == 0
#define	INPUT_FILE 			INPUT_FILE_0
#elif INPUT_IMAGE == 1
#define	INPUT_FILE 			INPUT_FILE_1
#elif INPUT_IMAGE == 2
#define	INPUT_FILE 			INPUT_FILE_2
#elif INPUT_IMAGE == 3
#define	INPUT_FILE 			INPUT_FILE_3
#elif INPUT_IMAGE == 4
#define	INPUT_FILE 			INPUT_FILE_4
#elif INPUT_IMAGE == 5
#define	INPUT_FILE 			INPUT_FILE_5
#elif INPUT_IMAGE == 8
#define	INPUT_FILE 			INPUT_FILE_8
#else // INPUT_IMAGE == 9
#define	INPUT_FILE 			INPUT_FILE_9
#endif

#define	INPUT_FILE_NAME			"Data/Input/" INPUT_FILE ".jpg"
#define	OUTPUT_FILE_NAME		"Data/Output/" INPUT_FILE "_out.png"
/////////////////////////////////////////////////////////////////////////////////////

#define MAKE_STAT_ITEM_LIST_CPU_CBO(string, time)  sprintf((string), "\n*** Host(CPU):: KER = %d, Time: %.3fms", \
			KERNEL_SELECTION, time);
#define MAKE_STAT_ITEM_LIST_CBO(string, gws, lws) sprintf((string), "IMAGE = %d(%s), GF_SIZE = %d, N_EXE = %d,\n " \
                    "    KER = %d, GWS = (%d, %d), LWS = (%d, %d)", INPUT_IMAGE, INPUT_FILE_NAME, GAUSSIAN_FILTER_SIZE, \
					N_EXECUTIONS, KERNEL_SELECTION, (gws)[0], (gws)[1], (lws)[0], (lws)[1]);