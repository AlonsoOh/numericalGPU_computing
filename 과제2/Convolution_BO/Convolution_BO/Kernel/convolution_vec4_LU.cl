//
//  convolution_vec4_LU.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

#define		FS_5		5	// Filter Size
#define		HFS_5		2	// Half Filter Size
__kernel void convolution_vec4_LU_5(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width) {

	int column = get_global_id(0);
	int row = get_global_id(1);

	float4 pixel, sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int2 cur_pixel_coord;

	int filter_index = 0;
	for (int r = -HFS_5; r <= HFS_5; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		{   // for (int c = -HFS_5; c <= HFS_5; c++) {
			int y_offset = cur_pixel_coord.y * n_columns;

			cur_pixel_coord.x = column - 2;		// -2
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4( *(input_data + y_offset + cur_pixel_coord.x) );
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// -1
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4( *(input_data + y_offset + cur_pixel_coord.x) );
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// 0
			pixel = convert_float4( *(input_data + y_offset + cur_pixel_coord.x) );
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// +1
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4( *(input_data + y_offset + cur_pixel_coord.x) );
			sum += pixel * filter_weights[filter_index++]; 

			cur_pixel_coord.x++;		// +2
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4( *(input_data + y_offset + cur_pixel_coord.x) );
			sum += pixel * filter_weights[filter_index++];  
		}
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////

#define		FS_7		7	// Filter Size
#define		HFS_7		3	// Half Filter Size
__kernel void convolution_vec4_LU_7(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width) {

	int column = get_global_id(0);
	int row = get_global_id(1);

	float4 pixel, sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int2 cur_pixel_coord;

	int filter_index = 0;
	for (int r = -HFS_7; r <= HFS_7; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		{   // for (int c = -HFS_7; c <= HFS_7; c++) {
			int y_offset = cur_pixel_coord.y * n_columns;

			cur_pixel_coord.x = column - 3;		// -3
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// -2
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// -1
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// 0
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// +1
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// +2
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// +3
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  
		}
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////

#define		FS_9		9	// Filter Size
#define		HFS_9		4	// Half Filter Size
__kernel void convolution_vec4_LU_9(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width) {

	int column = get_global_id(0);
	int row = get_global_id(1);

	float4 pixel, sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int2 cur_pixel_coord;

	int filter_index = 0;
	for (int r = -HFS_9; r <= HFS_9; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		{   // for (int c = -HFS_9; c <= HFS_9; c++) {
			int y_offset = cur_pixel_coord.y * n_columns;

			cur_pixel_coord.x = column - 4;		// -4
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];  

			cur_pixel_coord.x++;		// -3
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// -2
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// -1
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// 0
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +1
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +2
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +3
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +4
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];
		}
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////

#define		FS_11			11	// Filter Size
#define		HFS_11		5	// Half Filter Size
__kernel void convolution_vec4_LU_11(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width) {

	int column = get_global_id(0);
	int row = get_global_id(1);

	float4 pixel, sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int2 cur_pixel_coord;

	int filter_index = 0;
	for (int r = -HFS_11; r <= HFS_11; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		{   // for (int c = -HFS_11; c <= HFS_11; c++) {
			int y_offset = cur_pixel_coord.y * n_columns;

			cur_pixel_coord.x = column - 5;		// -5
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// -4
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// -3
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// -2
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// -1
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// 0
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +1
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +2
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +3
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		// +4
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];

			cur_pixel_coord.x++;		//+5
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			pixel = convert_float4(*(input_data + y_offset + cur_pixel_coord.x));
			sum += pixel * filter_weights[filter_index++];
		}
	}
	*(output_data + row * n_columns + column) = convert_uchar4(sum);
}