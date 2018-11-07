#include "common.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

template<typename Dtype>
__global__ void EdgeComputation_kernel(const int num_kernels, Dtype* input, Dtype* output, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;

		int window_size = 1;
		for (int m = -window_size; m <= window_size; m++) {
			for (int n = -window_size; n <= window_size; n++) {
				if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
					continue;
				int image_offset = (y + m) * width + x + n;
				*(output + point_offset) += fabs(*(input + point_offset)-*(input + image_offset));
			}
		}

		if (y-2 >= 0)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + (y - 2) * width + x));
		if (y+2 < height)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + (y + 2) * width + x));
		if (x-2 >= 0)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + y * width + x - 2));
		if (x+2 < width)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + y * width + x + 2));

		*(output + point_offset) = *(output + point_offset)/6;
	}
}

template<typename Dtype>
void EdgeComputation(cudaStream_t stream, Dtype* input, Dtype* output, int height, int width)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;
	EdgeComputation_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input, output, height, width);
}

template<typename Dtype>
__global__ void EdgeComputation_backward_kernel(const int num_kernels, Dtype* input, Dtype* gradOutput, Dtype* gradInput, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;

		int window_size = 1;
		for (int m = -window_size; m <= window_size; m++) {
			for (int n = -window_size; n <= window_size; n++) {
				if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
					continue;
				int image_offset = (y + m) * width + x + n;

				*(gradInput + point_offset) += (*(input + point_offset) > *(input + image_offset) ? 1 : -1) * *(gradOutput + point_offset);
				*(gradInput + point_offset) += (*(input + point_offset) > *(input + image_offset) ? 1 : -1) * *(gradOutput + image_offset);
			}
		}

		if (y-2 >= 0)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y - 2) * width + x) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y - 2) * width + x) ? 1 : -1) * *(gradOutput + (y - 2) * width + x);
		}
		if (y+2 < height)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y + 2) * width + x) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y + 2) * width + x) ? 1 : -1) * *(gradOutput + (y + 2) * width + x);
		}
		if (x-2 >= 0)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x - 2) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x - 2) ? 1 : -1) * *(gradOutput + y * width + x - 2);
		}
		if (x+2 < width)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x + 2) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x + 2) ? 1 : -1) * *(gradOutput + y * width + x + 2);
		}

		*(gradInput + point_offset) = *(gradInput + point_offset)/6;

	}
}

template<typename Dtype>
void EdgeComputation_backward(cudaStream_t stream, Dtype* input, Dtype* gradOutput, Dtype* gradInput, int height, int width)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;
	EdgeComputation_backward_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input, gradOutput, gradInput, height, width);
}