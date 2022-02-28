//
//  convolution_vec4.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

__kernel void convolution_vec4(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* filter_weights, int filter_width) {

	int column = get_global_id(0);
	int row = get_global_id(1);

	const int HFS = (int)(filter_width / 2); // Half Filter Size
	float4 pixel, sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int2 cur_pixel_coord;

	int filter_index = 0;
	for (int r = -HFS; r <= HFS; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		for (int c = -HFS; c <= HFS; c++) {
			cur_pixel_coord.x = column + c;
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;

			pixel = convert_float4( *(input_data + cur_pixel_coord.y * n_columns + cur_pixel_coord.x) );
			sum += pixel * filter_weights[filter_index++];
		}
	}
	*(output_data + row * n_columns + get_global_id(0)) = convert_uchar4(sum);
}