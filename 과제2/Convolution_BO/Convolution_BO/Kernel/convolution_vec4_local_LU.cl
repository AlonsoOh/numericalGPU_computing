//
//  convolution_vec4_local_LU.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

#define INPUT(x, y)				input_data[n_columns * (y) + (x)] 
#define SHARED_MEM(x, y)		shared_mem[SMW * (y) + (x)]

#define		FS_5		5	// Filter Size
#define		HFS_5		2	// Half Filter Size
void process_boundary_work_groups_LU_5(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_5;

	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	int x_coord, y_coord;
	if (loc_column < HFS_5) {
		x_coord = (column - HFS_5 < 0)? 0 : column - HFS_5;
		SHARED_MEM(loc_column, loc_row + HFS_5) = INPUT(x_coord, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_5) {
		x_coord = (column + HFS_5 >= n_columns)? n_columns - 1 : column + HFS_5;
		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5) = INPUT(x_coord, row);
		side_right = 1;
	}

	if (loc_row < HFS_5) {
		y_coord = (row - HFS_5 < 0)? 0 : row - HFS_5;
		SHARED_MEM(loc_column + HFS_5, loc_row) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_5 < 0)? 0 : column - HFS_5;
			y_coord = (row - HFS_5 < 0)? 0 : row - HFS_5;
			SHARED_MEM(loc_column, loc_row) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_5 >= n_columns) ? n_columns - 1 : column + HFS_5;
			y_coord = (row - HFS_5 < 0) ? 0 : row - HFS_5;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row) = INPUT(x_coord, y_coord);
		}
	}
	else if (loc_row >= get_local_size(1) - HFS_5) {
		y_coord = (row + HFS_5 >= n_rows) ? n_rows - 1 : row + HFS_5;
		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_5 < 0) ? 0 : column - HFS_5;
			y_coord = (row + HFS_5 >= n_rows) ? n_rows - 1 : row + HFS_5;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_5) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_5 >= n_columns) ? n_columns - 1 : column + HFS_5;
			y_coord = (row + HFS_5 >= n_rows) ? n_rows - 1 : row + HFS_5;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5) = INPUT(x_coord, y_coord);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_5; row <= loc_row + HFS_5; row++) {
		//for (int col = loc_column - HFS_5; col <= loc_column + HFS_5; col++) {
			//sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];
		//}
		int col = loc_column - HFS_5;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

__kernel void convolution_vec4_local_LU_5(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	if (get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1 ||
		get_group_id(1) == 0 || get_group_id(1) == get_num_groups(1) - 1) {
	  	process_boundary_work_groups_LU_5( input_data, output_data, n_columns, n_rows,  filter_weights,  filter_width,  shared_mem);
		return;
	}
	 
	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_5;
 
 	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	if (loc_column < HFS_5) {
	 	SHARED_MEM(loc_column, loc_row + HFS_5) = INPUT(column - HFS_5, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_5) {
		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5) = INPUT(column + HFS_5, row);
		side_right = 1;
	}

	if (loc_row < HFS_5) {
		SHARED_MEM(loc_column + HFS_5, loc_row) = INPUT(column, row - HFS_5);
		if (side_left == 1)  
			SHARED_MEM(loc_column, loc_row) = INPUT(column - HFS_5, row - HFS_5);
		if (side_right == 1) 
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row) = INPUT(column + HFS_5, row - HFS_5);
	}
	else if (loc_row >= get_local_size(1) - HFS_5) {
		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5) = INPUT(column, row + HFS_5);
		if (side_left == 1) 
			SHARED_MEM(loc_column, loc_row + 2 * HFS_5) = INPUT(column - HFS_5, row + HFS_5);
		if (side_right == 1)  
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5) = INPUT(column + HFS_5, row + HFS_5);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_5; row <= loc_row + HFS_5; row++) {
		int col = loc_column - HFS_5;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_5, row + HFS_5)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////

#define		FS_7		7	// Filter Size
#define		HFS_7		3	// Half Filter Size
void process_boundary_work_groups_LU_7(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_7;

	SHARED_MEM(loc_column + HFS_7, loc_row + HFS_7) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	int x_coord, y_coord;
	if (loc_column < HFS_7) {
		x_coord = (column - HFS_7 < 0) ? 0 : column - HFS_7;
		SHARED_MEM(loc_column, loc_row + HFS_7) = INPUT(x_coord, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_7) {
		x_coord = (column + HFS_7 >= n_columns) ? n_columns - 1 : column + HFS_7;
		SHARED_MEM(loc_column + 2 * HFS_7, loc_row + HFS_7) = INPUT(x_coord, row);
		side_right = 1;
	}

	if (loc_row < HFS_7) {
		y_coord = (row - HFS_7 < 0) ? 0 : row - HFS_7;
		SHARED_MEM(loc_column + HFS_7, loc_row) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_7 < 0) ? 0 : column - HFS_7;
			y_coord = (row - HFS_7 < 0) ? 0 : row - HFS_7;
			SHARED_MEM(loc_column, loc_row) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_7 >= n_columns) ? n_columns - 1 : column + HFS_7;
			y_coord = (row - HFS_7 < 0) ? 0 : row - HFS_7;
			SHARED_MEM(loc_column + 2 * HFS_7, loc_row) = INPUT(x_coord, y_coord);
		}
	}
	else if (loc_row >= get_local_size(1) - HFS_7) {
		y_coord = (row + HFS_7 >= n_rows) ? n_rows - 1 : row + HFS_7;
		SHARED_MEM(loc_column + HFS_7, loc_row + 2 * HFS_7) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_7 < 0) ? 0 : column - HFS_7;
			y_coord = (row + HFS_7 >= n_rows) ? n_rows - 1 : row + HFS_7;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_7) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_7 >= n_columns) ? n_columns - 1 : column + HFS_7;
			y_coord = (row + HFS_7 >= n_rows) ? n_rows - 1 : row + HFS_7;
			SHARED_MEM(loc_column + 2 * HFS_7, loc_row + 2 * HFS_7) = INPUT(x_coord, y_coord);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_7; row <= loc_row + HFS_7; row++) {
		//for (int col = loc_column - HFS_7; col <= loc_column + HFS_7; col++) {
		//	  sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];
		//}
		int col = loc_column - HFS_7;		// -3
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// +3
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

__kernel void convolution_vec4_local_LU_7(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	if (get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1 ||
		get_group_id(1) == 0 || get_group_id(1) == get_num_groups(1) - 1) {
		process_boundary_work_groups_LU_7(input_data, output_data, n_columns, n_rows, filter_weights, filter_width, shared_mem);
		return;
	}

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_7;

	SHARED_MEM(loc_column + HFS_7, loc_row + HFS_7) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	if (loc_column < HFS_7) {
		SHARED_MEM(loc_column, loc_row + HFS_7) = INPUT(column - HFS_7, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_7) {
		SHARED_MEM(loc_column + 2 * HFS_7, loc_row + HFS_7) = INPUT(column + HFS_7, row);
		side_right = 1;
	}

	if (loc_row < HFS_7) {
		SHARED_MEM(loc_column + HFS_7, loc_row) = INPUT(column, row - HFS_7);
		if (side_left == 1)
			SHARED_MEM(loc_column, loc_row) = INPUT(column - HFS_7, row - HFS_7);
		if (side_right == 1)
			SHARED_MEM(loc_column + 2 * HFS_7, loc_row) = INPUT(column + HFS_7, row - HFS_7);
	}
	else if (loc_row >= get_local_size(1) - HFS_7) {
		SHARED_MEM(loc_column + HFS_7, loc_row + 2 * HFS_7) = INPUT(column, row + HFS_7);
		if (side_left == 1)
			SHARED_MEM(loc_column, loc_row + 2 * HFS_7) = INPUT(column - HFS_7, row + HFS_7);
		if (side_right == 1)
			SHARED_MEM(loc_column + 2 * HFS_7, loc_row + 2 * HFS_7) = INPUT(column + HFS_7, row + HFS_7);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_7; row <= loc_row + HFS_7; row++) {
		int col = loc_column - HFS_7;		// -3
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];

		col++;		// +3
		sum += convert_float4(SHARED_MEM(col + HFS_7, row + HFS_7)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////

#define		FS_9		9	// Filter Size
#define		HFS_9		4	// Half Filter Size
void process_boundary_work_groups_LU_9(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_9;

	SHARED_MEM(loc_column + HFS_9, loc_row + HFS_9) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	int x_coord, y_coord;
	if (loc_column < HFS_9) {
		x_coord = (column - HFS_9 < 0) ? 0 : column - HFS_9;
		SHARED_MEM(loc_column, loc_row + HFS_9) = INPUT(x_coord, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_9) {
		x_coord = (column + HFS_9 >= n_columns) ? n_columns - 1 : column + HFS_9;
		SHARED_MEM(loc_column + 2 * HFS_9, loc_row + HFS_9) = INPUT(x_coord, row);
		side_right = 1;
	}

	if (loc_row < HFS_9) {
		y_coord = (row - HFS_9 < 0) ? 0 : row - HFS_9;
		SHARED_MEM(loc_column + HFS_9, loc_row) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_9 < 0) ? 0 : column - HFS_9;
			y_coord = (row - HFS_9 < 0) ? 0 : row - HFS_9;
			SHARED_MEM(loc_column, loc_row) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_9 >= n_columns) ? n_columns - 1 : column + HFS_9;
			y_coord = (row - HFS_9 < 0) ? 0 : row - HFS_9;
			SHARED_MEM(loc_column + 2 * HFS_9, loc_row) = INPUT(x_coord, y_coord);
		}
	}
	else if (loc_row >= get_local_size(1) - HFS_9) {
		y_coord = (row + HFS_9 >= n_rows) ? n_rows - 1 : row + HFS_9;
		SHARED_MEM(loc_column + HFS_9, loc_row + 2 * HFS_9) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_9 < 0) ? 0 : column - HFS_9;
			y_coord = (row + HFS_9 >= n_rows) ? n_rows - 1 : row + HFS_9;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_9) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_9 >= n_columns) ? n_columns - 1 : column + HFS_9;
			y_coord = (row + HFS_9 >= n_rows) ? n_rows - 1 : row + HFS_9;
			SHARED_MEM(loc_column + 2 * HFS_9, loc_row + 2 * HFS_9) = INPUT(x_coord, y_coord);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_9; row <= loc_row + HFS_9; row++) {
		//for (int col = loc_column - HFS_9; col <= loc_column + HFS_9; col++) {
		//	  sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];
		//}
		int col = loc_column - HFS_9;		// -4
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// -3
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +3
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +4
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

__kernel void convolution_vec4_local_LU_9(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	if (get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1 ||
		get_group_id(1) == 0 || get_group_id(1) == get_num_groups(1) - 1) {
		process_boundary_work_groups_LU_9(input_data, output_data, n_columns, n_rows, filter_weights, filter_width, shared_mem);
		return;
	}

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_9;

	SHARED_MEM(loc_column + HFS_9, loc_row + HFS_9) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	if (loc_column < HFS_9) {
		SHARED_MEM(loc_column, loc_row + HFS_9) = INPUT(column - HFS_9, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_9) {
		SHARED_MEM(loc_column + 2 * HFS_9, loc_row + HFS_9) = INPUT(column + HFS_9, row);
		side_right = 1;
	}

	if (loc_row < HFS_9) {
		SHARED_MEM(loc_column + HFS_9, loc_row) = INPUT(column, row - HFS_9);
		if (side_left == 1)
			SHARED_MEM(loc_column, loc_row) = INPUT(column - HFS_9, row - HFS_9);
		if (side_right == 1)
			SHARED_MEM(loc_column + 2 * HFS_9, loc_row) = INPUT(column + HFS_9, row - HFS_9);
	}
	else if (loc_row >= get_local_size(1) - HFS_9) {
		SHARED_MEM(loc_column + HFS_9, loc_row + 2 * HFS_9) = INPUT(column, row + HFS_9);
		if (side_left == 1)
			SHARED_MEM(loc_column, loc_row + 2 * HFS_9) = INPUT(column - HFS_9, row + HFS_9);
		if (side_right == 1)
			SHARED_MEM(loc_column + 2 * HFS_9, loc_row + 2 * HFS_9) = INPUT(column + HFS_9, row + HFS_9);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_9; row <= loc_row + HFS_9; row++) {
		int col = loc_column - HFS_9;		// -4
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// -3
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +3
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];

		col++;		// +4
		sum += convert_float4(SHARED_MEM(col + HFS_9, row + HFS_9)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////

#define		FS_11			11	// Filter Size
#define		HFS_11		5	// Half Filter Size
void process_boundary_work_groups_LU_11(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_11;

	SHARED_MEM(loc_column + HFS_11, loc_row + HFS_11) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	int x_coord, y_coord;
	if (loc_column < HFS_11) {
		x_coord = (column - HFS_11 < 0) ? 0 : column - HFS_11;
		SHARED_MEM(loc_column, loc_row + HFS_11) = INPUT(x_coord, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_11) {
		x_coord = (column + HFS_11 >= n_columns) ? n_columns - 1 : column + HFS_11;
		SHARED_MEM(loc_column + 2 * HFS_11, loc_row + HFS_11) = INPUT(x_coord, row);
		side_right = 1;
	}

	if (loc_row < HFS_11) {
		y_coord = (row - HFS_11 < 0) ? 0 : row - HFS_11;
		SHARED_MEM(loc_column + HFS_11, loc_row) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_11 < 0) ? 0 : column - HFS_11;
			y_coord = (row - HFS_11 < 0) ? 0 : row - HFS_11;
			SHARED_MEM(loc_column, loc_row) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_11 >= n_columns) ? n_columns - 1 : column + HFS_11;
			y_coord = (row - HFS_11 < 0) ? 0 : row - HFS_11;
			SHARED_MEM(loc_column + 2 * HFS_11, loc_row) = INPUT(x_coord, y_coord);
		}
	}
	else if (loc_row >= get_local_size(1) - HFS_11) {
		y_coord = (row + HFS_11 >= n_rows) ? n_rows - 1 : row + HFS_11;
		SHARED_MEM(loc_column + HFS_11, loc_row + 2 * HFS_11) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS_11 < 0) ? 0 : column - HFS_11;
			y_coord = (row + HFS_11 >= n_rows) ? n_rows - 1 : row + HFS_11;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_11) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS_11 >= n_columns) ? n_columns - 1 : column + HFS_11;
			y_coord = (row + HFS_11 >= n_rows) ? n_rows - 1 : row + HFS_11;
			SHARED_MEM(loc_column + 2 * HFS_11, loc_row + 2 * HFS_11) = INPUT(x_coord, y_coord);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_11; row <= loc_row + HFS_11; row++) {
		//for (int col = loc_column - HFS_11; col <= loc_column + HFS_11; col++) {
		//	  sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];
		//}
		int col = loc_column - HFS_11;		// -5
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -4
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -3
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +3
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +4
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +5
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

__kernel void convolution_vec4_local_LU_11(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	if (get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1 ||
		get_group_id(1) == 0 || get_group_id(1) == get_num_groups(1) - 1) {
		process_boundary_work_groups_LU_11(input_data, output_data, n_columns, n_rows, filter_weights, filter_width, shared_mem);
		return;
	}

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_11;

	SHARED_MEM(loc_column + HFS_11, loc_row + HFS_11) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	if (loc_column < HFS_11) {
		SHARED_MEM(loc_column, loc_row + HFS_11) = INPUT(column - HFS_11, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_11) {
		SHARED_MEM(loc_column + 2 * HFS_11, loc_row + HFS_11) = INPUT(column + HFS_11, row);
		side_right = 1;
	}

	if (loc_row < HFS_11) {
		SHARED_MEM(loc_column + HFS_11, loc_row) = INPUT(column, row - HFS_11);
		if (side_left == 1)
			SHARED_MEM(loc_column, loc_row) = INPUT(column - HFS_11, row - HFS_11);
		if (side_right == 1)
			SHARED_MEM(loc_column + 2 * HFS_11, loc_row) = INPUT(column + HFS_11, row - HFS_11);
	}
	else if (loc_row >= get_local_size(1) - HFS_11) {
		SHARED_MEM(loc_column + HFS_11, loc_row + 2 * HFS_11) = INPUT(column, row + HFS_11);
		if (side_left == 1)
			SHARED_MEM(loc_column, loc_row + 2 * HFS_11) = INPUT(column - HFS_11, row + HFS_11);
		if (side_right == 1)
			SHARED_MEM(loc_column + 2 * HFS_11, loc_row + 2 * HFS_11) = INPUT(column + HFS_11, row + HFS_11);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS_11; row <= loc_row + HFS_11; row++) {
		int col = loc_column - HFS_11;		// -5
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -4
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -3
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -2
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// -1
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// 0
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +1
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +2
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +3
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +4
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];

		col++;		// +5
		sum += convert_float4(SHARED_MEM(col + HFS_11, row + HFS_11)) * filter_weights[filter_index++];
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}