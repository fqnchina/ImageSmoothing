#include "common.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

template<typename Dtype>
__global__ void SmoothAndEdgeTerm_Loss_forward_main_kernel(const int num_kernels, Dtype* input_cnn, Dtype* target_yuv, Dtype* smooth_mask, Dtype* target_edge_mask, Dtype* weight, Dtype* output, float sigma_color, float sigma_space, int window_size, float lp, int height, int width, int w_L2) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;
		int window_length = window_size * 2 + 1;
		int image_length = height * width;

		for (int m = -window_size; m <= window_size; m++) {
			for (int n = -window_size; n <= window_size; n++) {

				if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
					continue;
				int image_offset = (y + m) * width + x + n;

				int window_offset = point_offset + ((m+window_size) * window_length + n+window_size) * image_length;
				Dtype* target_yuv_center = target_yuv + point_offset;
				Dtype* target_yuv_offset = target_yuv + image_offset;
				Dtype* input_cnn_center = input_cnn + point_offset;
				Dtype* input_cnn_offset = input_cnn + image_offset;

				if (*(smooth_mask + point_offset) == 0)
				{
					*(weight + window_offset) = exp((powf(fabs(*target_yuv_offset - *target_yuv_center),2) + powf(fabs(*(target_yuv_offset + image_length) - *(target_yuv_center + image_length)),2) + powf(fabs(*(target_yuv_offset + image_length * 2) - *(target_yuv_center + image_length * 2)),2)) * sigma_color);
					if (*(weight + window_offset) < 0.3)
						*(weight + window_offset) = 0;
				}else{
					*(weight + window_offset) = exp((m*m+n*n) * sigma_space);
				}

				for (int p = 0; p < 3; p++)
				{
					if (*(target_edge_mask + point_offset) == 0)
					{
						if (*(smooth_mask + point_offset) == 0)
							*(output + point_offset + image_length*p) += *(weight + window_offset) * powf(fabs(*(input_cnn_center + image_length*p) - *(input_cnn_offset + image_length*p)),lp);
						else
							*(output + point_offset + image_length*p) += *(weight + window_offset) * powf(fabs(*(input_cnn_center + image_length*p) - *(input_cnn_offset + image_length*p)),2) * w_L2;
					}
				}
			}
		}
	}
}

template<typename Dtype>
__global__ void SmoothAndEdgeTerm_Loss_forward_pre1_kernel(const int num_kernels, Dtype* input_edge, Dtype* target_edge, Dtype* smooth_mask_pre, int height, int width, int isDetailEnhancement) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;

		Dtype* input_center = input_edge + point_offset;
		Dtype* target_center = target_edge + point_offset;
		
		if (isDetailEnhancement == 1){
			if (*input_center - *target_center > 0)
				*(smooth_mask_pre + point_offset) = 1;
		}
		else{
			if (*target_center < 20 && *input_center - *target_center > 10)
				*(smooth_mask_pre + point_offset) = 1;
		}
		if (*target_center == 0 && *input_center - *target_center > 0)
			*(smooth_mask_pre + point_offset) = 1;
	}
}

template<typename Dtype>
__global__ void SmoothAndEdgeTerm_Loss_forward_pre2_kernel(const int num_kernels, Dtype* smooth_mask_pre, Dtype* smooth_mask, int height, int width, int isStylization) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;
		
		if (isStylization == 1){
			int window_size = 3;
			for (int m = -window_size; m <= window_size; m++) {
				for (int n = -window_size; n <= window_size; n++) {
					if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
						continue;
					int image_offset = (y + m) * width + x + n;

					if (*(smooth_mask_pre + image_offset) == 1)
					{
						*(smooth_mask + point_offset) = 1;
						break;
					}
				}
			}
		}

		if (*(smooth_mask_pre + point_offset) == 1)
			*(smooth_mask + point_offset) = 1;
	}
}


template<typename Dtype>
void SmoothAndEdgeTerm_Loss_forward(cudaStream_t stream, Dtype* input_cnn, Dtype* input_edge, Dtype* target_yuv, Dtype* target_edge, Dtype* target_edge_mask, Dtype* smooth_mask_pre, Dtype* smooth_mask, Dtype* weight, Dtype* output, float sigma_color, float sigma_space, int window_size, float lp, int height, int width, int isDetailEnhancement, int isStylization, int w_L2)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;

	SmoothAndEdgeTerm_Loss_forward_pre1_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input_edge, target_edge, smooth_mask_pre, height, width, isDetailEnhancement);
	SmoothAndEdgeTerm_Loss_forward_pre2_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, smooth_mask_pre, smooth_mask, height, width, isStylization);
	
	SmoothAndEdgeTerm_Loss_forward_main_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input_cnn, target_yuv, smooth_mask, target_edge_mask, weight, output, sigma_color, sigma_space, window_size, lp, height, width, w_L2);
}







template<typename Dtype>
__global__ void SmoothAndEdgeTerm_Loss_backward_kernel(const int num_kernels, Dtype* input_cnn, Dtype* smooth_mask, Dtype* target_edge_mask, Dtype* weight, Dtype* gradInput, float sigma_color, int window_size, float lp, int height, int width, int w_L2) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;
		int window_length = window_size * 2 + 1;
		int image_length = height * width;

		for (int m = -window_size; m <= window_size; m++) {
			for (int n = -window_size; n <= window_size; n++) {
				if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
					continue;
				int image_offset = (y + m) * width + x + n;

				int window_offset = image_offset + ((-m + window_size) * window_length - n + window_size) * image_length;
				Dtype* input_cnn_center = input_cnn + point_offset;
				Dtype* input_cnn_offset = input_cnn + image_offset;

				for (int p = 0; p < 3; p++)
				{
					if (*(target_edge_mask + point_offset) == 0)
					{
						if (*(smooth_mask + point_offset) == 0)
							*(gradInput + point_offset + image_length*p) += *(weight + window_offset) * lp * MIN(powf(fabs(*(input_cnn_center + image_length*p) - *(input_cnn_offset + image_length*p)),lp-1),1) * (*(input_cnn_center + image_length*p) > *(input_cnn_offset + image_length*p) ? 1 : -1);
						else
							*(gradInput + point_offset + image_length*p) += *(weight + window_offset) * 2 * MIN(powf(fabs(*(input_cnn_center + image_length*p) - *(input_cnn_offset + image_length*p)),2 - 1),1) * (*(input_cnn_center + image_length*p) > *(input_cnn_offset + image_length*p) ? 1 : -1) * w_L2;
					}
					if (*(target_edge_mask + image_offset) == 0)
					{
						if (*(smooth_mask + image_offset) == 0)
							*(gradInput + point_offset + image_length*p) += *(weight + window_offset) * lp * MIN(powf(fabs(*(input_cnn_center + image_length*p) - *(input_cnn_offset + image_length*p)),lp-1),1) * (*(input_cnn_center + image_length*p) > *(input_cnn_offset + image_length*p) ? 1 : -1);
						else
							*(gradInput + point_offset + image_length*p) += *(weight + window_offset) * 2 * MIN(powf(fabs(*(input_cnn_center + image_length*p) - *(input_cnn_offset + image_length*p)),2 - 1),1) * (*(input_cnn_center + image_length*p) > *(input_cnn_offset + image_length*p) ? 1 : -1) * w_L2;
					}
				}
			}
		}

	}
}

template<typename Dtype>
void SmoothAndEdgeTerm_Loss_backward(cudaStream_t stream, Dtype* input_cnn, Dtype* smooth_mask, Dtype* target_edge_mask, Dtype* weight, Dtype* gradInput, float sigma_color,int window_size, float lp, int height, int width, int w_L2)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;
	SmoothAndEdgeTerm_Loss_backward_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input_cnn, smooth_mask, target_edge_mask, weight, gradInput, sigma_color, window_size, lp, height, width, w_L2);
}