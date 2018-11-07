#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "EdgeComputation.h"

void THNN_CudaEdgeComputation_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output) {

	long batchSize = input->size[0];
	long plane = input->size[1];
	long height = input->size[2];
	long width = input->size[3];

	THCudaTensor *input_n = THCudaTensor_new(state);
	THCudaTensor *output_n = THCudaTensor_new(state);

	// For each elt in batch, do:
	for (int elt = 0; elt < batchSize; elt ++) {
		// Matrix mulitply per output:
		THCudaTensor_select(state, input_n, input, 0, elt);
		THCudaTensor_select(state, output_n, output, 0, elt);

		EdgeComputation(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_n),
				THCudaTensor_data(state, output_n),
				height, width);
	}

	THCudaTensor_free(state, input_n);
	THCudaTensor_free(state, output_n);
}

void THNN_CudaEdgeComputation_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput) {

	long batchSize = input->size[0];
	long plane = input->size[1];
	long height = input->size[2];
	long width = input->size[3];

	THCudaTensor *input_n = THCudaTensor_new(state);
	THCudaTensor *gradOutput_n = THCudaTensor_new(state);
	THCudaTensor *gradInput_n = THCudaTensor_new(state);

	// For each elt in batch, do:
	for (int elt = 0; elt < batchSize; elt ++) {
		// Matrix mulitply per output:
		THCudaTensor_select(state, input_n, input, 0, elt);
		THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);
		THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);

		EdgeComputation_backward(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_n),
				THCudaTensor_data(state, gradOutput_n),
				THCudaTensor_data(state, gradInput_n),
				height, width);
	}

	THCudaTensor_free(state, input_n);
	THCudaTensor_free(state, gradOutput_n);
	THCudaTensor_free(state, gradInput_n);
}