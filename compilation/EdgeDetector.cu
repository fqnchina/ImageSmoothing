#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "EdgeDetector.h"

void THNN_CudaEdgeDetector_updateOutput(THCState *state, THCudaTensor *input_image, THCudaTensor *input_edge, THCudaTensor *label_preserve, THCudaTensor *label_eliminate, int isSmoothing) {

	long batchSize = input_image->size[0];
	long plane = input_image->size[1];
	long height = input_image->size[2];
	long width = input_image->size[3];

	THCudaTensor *input_image_n = THCudaTensor_new(state);
	THCudaTensor *input_edge_n = THCudaTensor_new(state);
	THCudaTensor *label_preserve_n = THCudaTensor_new(state);
	THCudaTensor *label_eliminate_n = THCudaTensor_new(state);

	for (int elt = 0; elt < batchSize; elt ++) {
		THCudaTensor_select(state, input_image_n, input_image, 0, elt);
		THCudaTensor_select(state, input_edge_n, input_edge, 0, elt);
		THCudaTensor_select(state, label_preserve_n, label_preserve, 0, elt);
		THCudaTensor_select(state, label_eliminate_n, label_eliminate, 0, elt);

		EdgeDetector(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_image_n),
				THCudaTensor_data(state, input_edge_n),
				THCudaTensor_data(state, label_preserve_n),
				THCudaTensor_data(state, label_eliminate_n),
				height, width, isSmoothing);
	}

	THCudaTensor_free(state, input_image_n);
	THCudaTensor_free(state, input_edge_n);
	THCudaTensor_free(state, label_preserve_n);
	THCudaTensor_free(state, label_eliminate_n);
}