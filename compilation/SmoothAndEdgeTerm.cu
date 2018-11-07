#include "THCUNN.h"
#include "common.h"
#include "im2col.h"
#include "SmoothAndEdgeTerm.h"

void THNN_CudaSmoothAndEdgeTerm_updateOutput(THCState *state, THCudaTensor *input_cnn, THCudaTensor *input_edge, THCudaTensor *target_yuv, THCudaTensor *target_edge, THCudaTensor *target_edge_mask, THCudaTensor *smooth_mask_pre, THCudaTensor *smooth_mask, THCudaTensor *weight, THCudaTensor *output, float sigma_color, float sigma_space, int window_size, float lp, int isDetailEnhancement, int isStylization, int w_L2) {

	long batchSize = input_cnn->size[0];
	long plane = input_cnn->size[1];
	long height = input_cnn->size[2];
	long width = input_cnn->size[3];

	THCudaTensor_resize4d(state, weight, batchSize, (window_size*2+1) * (window_size*2+1),
			height, width);
	THCudaTensor_fill(state, weight, 0);

	THCudaTensor *input_cnn_n = THCudaTensor_new(state);
	THCudaTensor *input_edge_n = THCudaTensor_new(state);
	THCudaTensor *target_yuv_n = THCudaTensor_new(state);
	THCudaTensor *target_edge_n = THCudaTensor_new(state);
	THCudaTensor *target_edge_mask_n = THCudaTensor_new(state);
	THCudaTensor *smooth_mask_pre_n = THCudaTensor_new(state);
	THCudaTensor *smooth_mask_n = THCudaTensor_new(state);
	THCudaTensor *weight_n = THCudaTensor_new(state);
	THCudaTensor *output_n = THCudaTensor_new(state);

	for (int elt = 0; elt < batchSize; elt++) {
		// Matrix mulitply per output:
		THCudaTensor_select(state, input_cnn_n, input_cnn, 0, elt);
		THCudaTensor_select(state, input_edge_n, input_edge, 0, elt);
		THCudaTensor_select(state, target_yuv_n, target_yuv, 0, elt);
		THCudaTensor_select(state, target_edge_n, target_edge, 0, elt);
		THCudaTensor_select(state, target_edge_mask_n, target_edge_mask, 0, elt);
		THCudaTensor_select(state, smooth_mask_pre_n, smooth_mask_pre, 0, elt);
		THCudaTensor_select(state, smooth_mask_n, smooth_mask, 0, elt);
		THCudaTensor_select(state, weight_n, weight, 0, elt);
		THCudaTensor_select(state, output_n, output, 0, elt);

		SmoothAndEdgeTerm_Loss_forward(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_cnn_n),
				THCudaTensor_data(state, input_edge_n),
				THCudaTensor_data(state, target_yuv_n),
				THCudaTensor_data(state, target_edge_n),
				THCudaTensor_data(state, target_edge_mask_n),
				THCudaTensor_data(state, smooth_mask_pre_n),
				THCudaTensor_data(state, smooth_mask_n),
				THCudaTensor_data(state, weight_n),
				THCudaTensor_data(state, output_n),
				sigma_color, sigma_space, window_size, lp, height, width, isDetailEnhancement, isStylization, w_L2);
	}

	// Free
	THCudaTensor_free(state, input_cnn_n);
	THCudaTensor_free(state, input_edge_n);
	THCudaTensor_free(state, target_yuv_n);
	THCudaTensor_free(state, target_edge_n);
	THCudaTensor_free(state, target_edge_mask_n);
	THCudaTensor_free(state, smooth_mask_pre_n);
	THCudaTensor_free(state, smooth_mask_n);
	THCudaTensor_free(state, weight_n);
	THCudaTensor_free(state, output_n);
}

void THNN_CudaSmoothAndEdgeTerm_updateGradInput(THCState *state, THCudaTensor *input_cnn, THCudaTensor *smooth_mask, THCudaTensor *target_edge_mask, THCudaTensor *weight, THCudaTensor *gradInput, float sigma_color, int window_size, float lp, int w_L2) {

	long batchSize = input_cnn->size[0];
	long plane = input_cnn->size[1];
	long height = input_cnn->size[2];
	long width = input_cnn->size[3];

	THCudaTensor *input_cnn_n = THCudaTensor_new(state);
	THCudaTensor *smooth_mask_n = THCudaTensor_new(state);
	THCudaTensor *target_edge_mask_n = THCudaTensor_new(state);
	THCudaTensor *weight_n = THCudaTensor_new(state);
	THCudaTensor *gradInput_n = THCudaTensor_new(state);

	for (int elt = 0; elt < batchSize; elt++) {
		THCudaTensor_select(state, input_cnn_n, input_cnn, 0, elt);
		THCudaTensor_select(state, smooth_mask_n, smooth_mask, 0, elt);
		THCudaTensor_select(state, target_edge_mask_n, target_edge_mask, 0, elt);
		THCudaTensor_select(state, weight_n, weight, 0, elt);
		THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);

		SmoothAndEdgeTerm_Loss_backward(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_cnn_n),
				THCudaTensor_data(state, smooth_mask_n),
				THCudaTensor_data(state, target_edge_mask_n),
				THCudaTensor_data(state, weight_n),
				THCudaTensor_data(state, gradInput_n),
				sigma_color, window_size, lp, height, width, w_L2);
	}

	// Free
	THCudaTensor_free(state, input_cnn_n);
	THCudaTensor_free(state, smooth_mask_n);
	THCudaTensor_free(state, target_edge_mask_n);
	THCudaTensor_free(state, weight_n);
	THCudaTensor_free(state, gradInput_n);
}
