//
//  convolution_vec4_local.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

#define INPUT(x, y)				input_data[n_columns * (y) + (x)] 
#define SHARED_MEM(x, y)		shared_mem[SMW * (y) + (x)]

void process_boundary_work_groups(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int HFS = (int)(filter_width / 2), SMW = get_local_size(0) + 2 * HFS;

	SHARED_MEM(loc_column + HFS, loc_row + HFS) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	int x_coord, y_coord;
	if (loc_column < HFS) {
		x_coord = (column - HFS < 0)? 0 : column - HFS;
		SHARED_MEM(loc_column, loc_row + HFS) = INPUT(x_coord, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS) {
		x_coord = (column + HFS >= n_columns)? n_columns - 1 : column + HFS;
		SHARED_MEM(loc_column + 2 * HFS, loc_row + HFS) = INPUT(x_coord, row);
		side_right = 1;
	}

	if (loc_row < HFS) {
		y_coord = (row - HFS < 0)? 0 : row - HFS;
		SHARED_MEM(loc_column + HFS, loc_row) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS < 0)? 0 : column - HFS;
			y_coord = (row - HFS < 0)? 0 : row - HFS;
			SHARED_MEM(loc_column, loc_row) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS >= n_columns) ? n_columns - 1 : column + HFS;
			y_coord = (row - HFS < 0) ? 0 : row - HFS;
			SHARED_MEM(loc_column + 2 * HFS, loc_row) = INPUT(x_coord, y_coord);
		}
	}
	else if (loc_row >= get_local_size(1) - HFS) {
		y_coord = (row + HFS >= n_rows) ? n_rows - 1 : row + HFS;
		SHARED_MEM(loc_column + HFS, loc_row + 2 * HFS) = INPUT(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS < 0) ? 0 : column - HFS;
			y_coord = (row + HFS >= n_rows) ? n_rows - 1 : row + HFS;
			SHARED_MEM(loc_column, loc_row + 2 * HFS) = INPUT(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS >= n_columns) ? n_columns - 1 : column + HFS;
			y_coord = (row + HFS >= n_rows) ? n_rows - 1 : row + HFS;
			SHARED_MEM(loc_column + 2 * HFS, loc_row + 2 * HFS) = INPUT(x_coord, y_coord);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS; row <= loc_row + HFS; row++)
		for (int col = loc_column - HFS; col <= loc_column + HFS; col++) {
			sum += convert_float4(SHARED_MEM(col + HFS, row + HFS)) * filter_weights[filter_index++];
		}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

__kernel void convolution_vec4_local(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width, __local uchar4* shared_mem) {

	if (get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1 ||
		get_group_id(1) == 0 || get_group_id(1) == get_num_groups(1) - 1) {
	  	process_boundary_work_groups( input_data, output_data, n_columns, n_rows,  filter_weights,  filter_width,  shared_mem);
		return;
	}
	 
	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int HFS = (int)(filter_width / 2), SMW = get_local_size(0) + 2 * HFS;
 
 	SHARED_MEM(loc_column + HFS, loc_row + HFS) = INPUT(column, row);

	int side_left = 0, side_right = 0;
	if (loc_column < HFS) {
	 	SHARED_MEM(loc_column, loc_row + HFS) = INPUT(column - HFS, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS) {
		SHARED_MEM(loc_column + 2 * HFS, loc_row + HFS) = INPUT(column + HFS, row);
		side_right = 1;
	}

	if (loc_row < HFS) {
		SHARED_MEM(loc_column + HFS, loc_row) = INPUT(column, row - HFS);
		if (side_left == 1)  
			SHARED_MEM(loc_column, loc_row) = INPUT(column - HFS, row - HFS);
		if (side_right == 1) 
			SHARED_MEM(loc_column + 2 * HFS, loc_row) = INPUT(column + HFS, row - HFS);
	}
	else if (loc_row >= get_local_size(1) - HFS) {
		SHARED_MEM(loc_column + HFS, loc_row + 2 * HFS) = INPUT(column, row + HFS);
		if (side_left == 1) 
			SHARED_MEM(loc_column, loc_row + 2 * HFS) = INPUT(column - HFS, row + HFS);
		if (side_right == 1)  
			SHARED_MEM(loc_column + 2 * HFS, loc_row + 2 * HFS) = INPUT(column + HFS, row + HFS);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sum = (float4)(0.01f, 0.01f, 0.01f, 0.01f);
	int filter_index = 0;
	for (int row = loc_row - HFS; row <= loc_row + HFS; row++)  
		for (int col = loc_column - HFS; col <= loc_column + HFS; col++) {
			sum += convert_float4(SHARED_MEM(col + HFS, row + HFS)) * filter_weights[filter_index++];
		}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}