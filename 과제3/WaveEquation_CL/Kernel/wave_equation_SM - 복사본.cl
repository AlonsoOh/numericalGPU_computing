
#define SM_ADDR(i, j) ( (j) * (LSZ_0 + 2) + (i) )

#define LID_0	get_local_id(0)
#define LID_1	get_local_id(1)

#define GID_0	get_global_id(0)
#define GID_1	get_global_id(1)

#define LSZ_0	get_local_size(0)
#define LSZ_1	get_local_size(1)

#define GSZ_0	get_global_size(0)
#define GSZ_1	get_global_size(1)

__kernel void wave_equation_SM(
	__global float4* grid_next,
	__global float4* grid_cur, 
 	__global float4* grid_prev,
	float beta, float diag_el_of_A, 
 	__local float* shared_grid_block) {


	int index = GID_1 * GSZ_0 + GID_0;

	
	shared_grid_block[SM_ADDR(LID_0 + 1, LID_1 + 1)] = grid_cur[index].y;


	if (LID_0 == 0)  { // left
		if (GID_0 == 0) 
			shared_grid_block[SM_ADDR(0, LID_1 + 1)] = 0;
		else 
			shared_grid_block[SM_ADDR(0, LID_1 + 1)] = grid_cur[index - 1].y;
		 	


}
 

	if (LID_0 == LSZ_0 - 1) { // right
		if (GID_0 == get_num_groups(0) - 1)
			shared_grid_block[SM_ADDR(LID_0 + 2, LID_1 + 1)] = 0;
		else 
			shared_grid_block[SM_ADDR(LID_0 + 2, LID_1 + 1)] = grid_cur[index + 1].y;



		}

	if (LID_1 == 0) { // up
		if (GID_1 == 0)
			shared_grid_block[SM_ADDR(LID_0 + 1, 0)] = 0;
		else
			shared_grid_block[SM_ADDR(LID_0 + 1, 0)] = grid_cur[index - GSZ_0].y;

		}
 
	if (LID_1 == LSZ_1 - 1) { // down
		if (GID_1 == get_num_groups(1) - 1)
			shared_grid_block[SM_ADDR(LID_0 + 1, LID_1 + 2)] = 0;
		else 
			shared_grid_block[SM_ADDR(LID_0 + 1, LID_1 + 2)] = grid_cur[index + GSZ_0].y;

  }

	if ( (LID_0 == 0) && (LID_1 == 0) )  { // left & up
		if ( (GID_0 == 0) && (GID_1 == 0) )
			shared_grid_block[SM_ADDR(0, 0)] = 0;
		else
		    shared_grid_block[SM_ADDR(0, 0)] = grid_cur[index - GSZ_0 - 1].y;
		}

	if ( (LID_0 == LSZ_0 - 1) && (LID_1 == 0) ) {  // right & up 
		if ( (GID_0 == get_num_groups(0) - 1) && (GID_1 == 0) ) 
			shared_grid_block[SM_ADDR(LSZ_0 + 1, 0)] = 0;
		else
			shared_grid_block[SM_ADDR(LSZ_0 + 1, 0)] = grid_cur[index - GSZ_0 + 1].y;
		}

	if ( (LID_0 == 0) && (LID_1 == LSZ_1 - 1) ) { // left & down
		if ( (GID_0 == 0) && (GID_1 == get_num_groups(1) - 1) ) 
		 shared_grid_block[SM_ADDR(0, LSZ_1 + 1)] = 0;
		else 
			shared_grid_block[SM_ADDR(0, LSZ_1 + 1)] = grid_cur[index + GSZ_0 - 1].y;
		}

	if ( (LID_0 == LSZ_0 - 1) && (LID_1 == LSZ_1 - 1) ) { // right & down
		if ( (GID_0 == get_num_groups(0) - 1) && (GID_1 == get_num_groups(1) - 1) )
			shared_grid_block[SM_ADDR(LSZ_0 + 1, LSZ_1 + 1)] = 0;
		else
			shared_grid_block[SM_ADDR(LSZ_0 + 1, LSZ_1 + 1)] = grid_cur[index + GSZ_0 + 1].y;
		}

 barrier(CLK_LOCAL_MEM_FENCE);


    	if ( (get_global_id(0) == 0) || (get_global_id(0) == GSZ_0 - 1) ) return;
     if ( (get_global_id(1) == 0) || (get_global_id(1) == get_global_size(1) - 1) ) return;

 
//	float sum = grid_cur[index - GSZ_0].y + grid_cur[index - 1].y 
//	              + grid_cur[index + 1].y + grid_cur[index + GSZ_0].y;

	float sum = shared_grid_block[SM_ADDR(LID_0 + 1, LID_1)]
					+ shared_grid_block[SM_ADDR(LID_0, LID_1 + 1)]
					+ shared_grid_block[SM_ADDR(LID_0 + 2, LID_1 + 1)]
					+ shared_grid_block[SM_ADDR(LID_0 + 1, LID_1 + 2)];

//  printf("%d %d %f\n", get_global_id(0), get_global_id(1), sum);
 
 
 	grid_next[index].y = (2 * shared_grid_block[SM_ADDR(LID_0 + 1, LID_1 + 1)]
						- grid_prev[index].y + beta * sum) / diag_el_of_A;

						/*

//	grid_next[index].y = (2 * grid_cur[index].y  - grid_prev[index].y + beta * sum) / diag_el_of_A;

*/
}
