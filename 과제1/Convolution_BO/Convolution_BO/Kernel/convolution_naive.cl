//
//  convolution_naive.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//
__kernel void convolution_naive(
	const __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width) {

	int column = get_global_id(0);
	int row = get_global_id(1);

	const int HFS = (int)(filter_width / 2); // Half Filter Size
	float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	int2 cur_pixel_coord;
	uchar pixel[4];
	int offset;

	int filter_index = 0;
	for (int r = -HFS; r <= HFS; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		for (int c = -HFS; c <= HFS; c++) {
			cur_pixel_coord.x = column + c;
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			
			offset = 4 * (cur_pixel_coord.y * n_columns + cur_pixel_coord.x);
			for (int i = 0; i < 4; i++)  
				pixel[i] = *(input_data + offset + i);
			for (int i = 0; i < 4; i++)  
				sum[i] += ((float) pixel[i]) * filter_weights[filter_index];
			filter_index++;
		}
	}
	offset = 4 * (row * n_columns + column);
	for (int i = 0; i < 4; i++)
		*(output_data + offset + i) = (uchar)sum[i];
}