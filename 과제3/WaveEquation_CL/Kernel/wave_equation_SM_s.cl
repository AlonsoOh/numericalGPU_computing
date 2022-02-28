
#define SM_ADDR(i, j) ( (j) * (local_size_0 + 2) + (i) )

__kernel void wave_equation_SM(
	__global float4* grid_next,
	__global float4* grid_cur, 
 	__global float4* grid_prev,
	float beta, float diag_el_of_A, 
 	__local float* shared_grid_block) {


	 int local_id_0 = get_local_id(0);
	int local_id_1 = get_local_id(1);
	int local_size_0 = get_local_size(0);
	int local_size_1 = get_local_size(1);
	int global_id_0 = get_global_id(0);
	int global_id_1 = get_global_id(1);


	int global_size_0 = get_global_size(0);
		int global_size_1 = get_global_size(1);

	int index = get_global_id(1) * global_size_0 + get_global_id(0);

	
	shared_grid_block[SM_ADDR(local_id_0 + 1, local_id_1 + 1)] = grid_cur[index].y;


	if (local_id_0 == 0)  { // left
		if (global_id_0 == 0) 
			shared_grid_block[SM_ADDR(0, local_id_1 + 1)] = 0;
		else 
			shared_grid_block[SM_ADDR(0, local_id_1 + 1)] = grid_cur[index - 1].y;
		 	


}
 

	if (local_id_0 == local_size_0 - 1) { // right
		if (global_id_0 == global_size_0 - 1)
			shared_grid_block[SM_ADDR(local_id_0 + 2, local_id_1 + 1)] = 0;
		else 
			shared_grid_block[SM_ADDR(local_id_0 + 2, local_id_1 + 1)] = grid_cur[index + 1].y;



		}

	if (local_id_1 == 0) { // up
		if (global_id_1 == 0)
			shared_grid_block[SM_ADDR(local_id_0 + 1, 0)] = 0;
		else
			shared_grid_block[SM_ADDR(local_id_0 + 1, 0)] = grid_cur[index - global_size_0].y;

		}
 
	if (local_id_1 == local_size_1 - 1) { // down
		if (global_id_1 == global_size_1 - 1)
			shared_grid_block[SM_ADDR(local_id_0 + 1, local_id_1 + 2)] = 0;
		else 
			shared_grid_block[SM_ADDR(local_id_0 + 1, local_id_1 + 2)] = grid_cur[index + global_size_0].y;

  }

	if ( (local_id_0 == 0) && (local_id_1 == 0) )  { // left & up
		if ( (global_id_0 == 0) && (global_id_1 == 0) )
			shared_grid_block[SM_ADDR(0, 0)] = 0;
		else
		    shared_grid_block[SM_ADDR(0, 0)] = grid_cur[index - global_size_0 - 1].y;
		}

	if ( (local_id_0 == local_size_0 - 1) && (local_id_1 == 0) ) {  // right & up 
		if ( (global_id_0 == global_size_0 - 1) && (global_id_1 == 0) ) 
			shared_grid_block[SM_ADDR(local_size_0 + 1, 0)] = 0;
		else
			shared_grid_block[SM_ADDR(local_size_0 + 1, 0)] = grid_cur[index - global_size_0 + 1].y;
		}

	if ( (local_id_0 == 0) && (local_id_1 == local_size_1 - 1) ) { // left & down
		if ( (global_id_0 == 0) && (global_id_1 == global_size_1 - 1) ) 
		 shared_grid_block[SM_ADDR(0, local_size_1 + 1)] = 0;
		else 
			shared_grid_block[SM_ADDR(0, local_size_1 + 1)] = grid_cur[index + global_size_0 - 1].y;
		}

	if ( (local_id_0 == local_size_0 - 1) && (local_id_1 == local_size_1 - 1) ) { // right & down
		if ( (global_id_0 == global_size_0 - 1) && (global_id_1 == global_size_1 - 1) )
			shared_grid_block[SM_ADDR(local_size_0 + 1, local_size_1 + 1)] = 0;
		else
			shared_grid_block[SM_ADDR(local_size_0 + 1, local_size_1 + 1)] = grid_cur[index + global_size_0 + 1].y;
		}

 
 
 barrier(CLK_LOCAL_MEM_FENCE);

 /*
		 if ( (local_id_0 == 1) && ( local_id_1 == 1) ) {
 for (int i = 0 ; i < local_size_0 + 2; i++) {
   for (int j = 0; j < local_size_1 + 2; j++) {
  printf(" (%d, %d), %d %f ", i, j, aaa[SM_ADDR(i, j)], shared_grid_block[SM_ADDR(i, j)]);
 }
 printf("\n****\n");
 }
 }
 return;
 */

    	if ( (get_global_id(0) == 0) || (get_global_id(0) == global_size_0 - 1) ) return;
     if ( (get_global_id(1) == 0) || (get_global_id(1) == get_global_size(1) - 1) ) return;

 
//	float sum = grid_cur[index - global_size_0].y + grid_cur[index - 1].y 
//	              + grid_cur[index + 1].y + grid_cur[index + global_size_0].y;

	float sum = shared_grid_block[SM_ADDR(local_id_0 + 1, local_id_1)]
					+ shared_grid_block[SM_ADDR(local_id_0, local_id_1 + 1)]
					+ shared_grid_block[SM_ADDR(local_id_0 + 2, local_id_1 + 1)]
					+ shared_grid_block[SM_ADDR(local_id_0 + 1, local_id_1 + 2)];

//  printf("%d %d %f\n", get_global_id(0), get_global_id(1), sum);
 
 
 	grid_next[index].y = (2 * shared_grid_block[SM_ADDR(local_id_0 + 1, local_id_1 + 1)]
						- grid_prev[index].y + beta * sum) / diag_el_of_A;

						/*

//	grid_next[index].y = (2 * grid_cur[index].y  - grid_prev[index].y + beta * sum) / diag_el_of_A;

*/
}
