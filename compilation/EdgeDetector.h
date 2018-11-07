#include "common.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define PI 3.1415926535897932384626433832

template<typename Dtype>
__global__ void EdgeSelection_kernel(const int num_kernels, Dtype* input_edge, Dtype* output, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;

		Dtype* input_edge_center = input_edge + point_offset;
		if (*input_edge_center > 25 && *input_edge_center < 60)
		{
			int sharpCount = 0;
			int smoothCount = 0;

			int window_size = 10;
			for (int m = -window_size; m <= window_size; m++) {
				for (int n = -window_size; n <= window_size; n++) {
					if (m == 0 && n == 0)
						continue;
					if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
						continue;

					int image_offset = (y + m) * width + x + n;
					Dtype* input_edge_offset = input_edge + image_offset;
					if (*input_edge_center - *input_edge_offset > 20)
						smoothCount++;
				}
			}
			if (smoothCount > 200)
				*(output + point_offset) = 1;
		}

		if (*input_edge_center >= 60)
		{
			int sharpCount = 0;
			int smoothCount = 0;

			int window_size = 10;
			for (int m = -window_size; m <= window_size; m++) {
				for (int n = -window_size; n <= window_size; n++) {
					if (m == 0 && n == 0)
						continue;
					if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
						continue;

					int image_offset = (y + m) * width + x + n;
					Dtype* input_edge_offset = input_edge + image_offset;
					if (*input_edge_center - *input_edge_offset > 30)
						smoothCount++;
				}
			}
			if (smoothCount > 200)
				*(output + point_offset) = 1;
		}

	}
}

template<typename Dtype>
__global__ void EdgeFiltering_kernel(const int num_kernels, Dtype* input_edge, Dtype* output, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;
		
		Dtype* output_center = output + point_offset;
		if(*output_center == 1)
		{
			int thres = 20;
			int count = 0;
			int countAll = 0;

			// horizontal 
			int window_size = 25;
			int temp_x = x;
			for (int m = 1; m <= window_size; m++)
			{
				if (y+m < 0 || y+m >= height)
					continue;
				int precount = count;
				for (int n = -1; n <= 1; n++)
				{
					if (temp_x+n < 0 || temp_x+n >= width)
						continue;
					Dtype* output_offset = output + (y + m) * width + temp_x + n;
					if (*output_offset == 1)
					{
						temp_x = temp_x + n;
						count++;
						break;
					}
				}
				if(precount == count)
					break;
			}	

			temp_x = x;
			for (int m = -1; m >= -window_size; m--)
			{
				if (y+m < 0 || y+m >= height)
					continue;
				int precount = count;
				for (int n = -1; n <= 1; n++)
				{
					if(temp_x+n < 0 || temp_x+n >= width)
						continue;
					Dtype* output_offset = output + (y + m) * width + temp_x + n;
					if (*output_offset == 1)
					{
						temp_x = temp_x + n;
						count++;
						break;
					}
				}
				if(precount == count)
					break;
			}
			if (count < thres)
				countAll++;


			//vertical
			count = 0;
			int temp_y = y;
			for (int n = 1; n <= window_size; n++)
			{
				if (x+n < 0 || x+n >= width)
					continue;
				int precount = count;
				for (int m = -1; m <= 1; m++)
				{
					if(temp_y+m < 0 || temp_y+m >= height)
						continue;
					Dtype* output_offset = output + (temp_y + m) * width + x + n;
					if (*output_offset == 1)
					{
						temp_y = temp_y + m;
						count++;
						break;
					}
				}
				if(precount == count)
					break;
			}

			temp_y = y;
			for (int n = -1; n >= -window_size; n--)
			{
				if (x+n < 0 || x+n >= width)
					continue;
				int precount = count;
				for (int m = -1; m <= 1; m++)
				{
					if(temp_y+m < 0 || temp_y+m >= height)
						continue;
					Dtype* output_offset = output + (temp_y + m) * width + x + n;
					if (*output_offset == 1)
					{
						temp_y = temp_y + m;
						count++;
						break;
					}
				}
				if(precount == count)
					break;
			}
			if (count < thres)
				countAll++;



			//diagonal
			count = 0;
			temp_x = x, temp_y = y;
			for (int p = 1; p <= window_size; p++)
			{
				int m = 0, n = 1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				Dtype* output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}

				m = 1, n = 0;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}

				m = 1, n = 1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}
				break;
			}

			temp_x = x, temp_y = y;
			for (int p = 1; p <= window_size; p++)
			{
				int m = 0, n = -1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				Dtype* output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}

				m = -1, n = 0;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}
				
				m = -1, n = -1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}
				break;
			}
			if (count < thres)
				countAll++;



			//diagonal -1
			count = 0;
			temp_x = x, temp_y = y;
			for (int p = 1; p <= window_size; p++)
			{
				int m = 0, n = 1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				Dtype* output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}

				m = -1, n = 0;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}

				m = -1, n = 1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}
				break;
			}

			temp_x = x, temp_y = y;
			for (int p = 1; p <= window_size; p++)
			{
				int m = 0, n = -1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				Dtype* output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}

				m = 1, n = 0;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}
				
				m = 1, n = -1;
				if(temp_y+m < 0 || temp_y+m >= height || temp_x+n < 0 || temp_x+n >= width)
					continue;
				output_offset = output + (temp_y + m) * width + temp_x + n;
				if (*output_offset == 1)
				{
					temp_y = temp_y + m;
					temp_x = temp_x + n;
					count++;
					continue;
				}
				break;
			}
			if (count <= thres)
				countAll++;

			if (countAll == 4)
				*output_center = 0;
		}
	}
}

template<typename Dtype>
__global__ void EdgeTexture_kernel(const int num_kernels, Dtype* input_image, Dtype* input_edge, Dtype* output, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;
		int image_length = height*width;

		Dtype* input_image_center = input_image + point_offset;
		Dtype* input_edge_center = input_edge + point_offset;
		if (*input_edge_center > 25)
		{

			if (y-1 >= 0 && y+1 < height && x-1 >= 0 && x+1 < width)
			{
				double y_off = 2 * (*(input_image + (y + 1) * width + x) - *(input_image + (y - 1) * width + x)) + (*(input_image + (y + 1) * width + x + 1) - *(input_image + (y - 1) * width + x + 1)) + (*(input_image + (y + 1) * width + x - 1) - *(input_image + (y - 1) * width + x - 1));
				double x_off = 2 * (*(input_image + y * width + x + 1) - *(input_image + y * width + x - 1)) + (*(input_image + (y+1) * width + x + 1) - *(input_image + (y+1) * width + x - 1)) + (*(input_image + (y-1) * width + x + 1) - *(input_image + (y-1) * width + x - 1));

				double angle = 0;
				if (x_off == 0){
					if (y_off > 0)
						angle = PI / 2;
					else if (y_off <= 0)
						angle = PI*1.5;
				}
				else if (y_off == 0){
					if (x_off > 0)
						angle = 0;
					else if (x_off <= 0)
						angle = PI;
				}
				else
					angle = atan2(y_off, x_off);

				if (angle < 0)
					angle += PI;

				int point_dis = 2;
				int a_x = x + point_dis*cos(angle); 
				int a_y = y + point_dis*sin(angle);
				int b_x = x + point_dis*cos(angle + PI); 
				int b_y = y + point_dis*sin(angle + PI);

				double averageA = 0;
				int countA = 0;
				int window_size = 1;
				for (int m = -window_size; m <= window_size; m++) {
					for (int n = -window_size; n <= window_size; n++) {
						if (m == 0 && n == 0)
							continue;
						if (a_y+m < 0 || a_y+m >= height || a_x+n < 0 || a_x+n >= width)
							continue;

						int image_offset = (a_y + m) * width + a_x + n;
						Dtype* input_image_offset = input_image + image_offset;
						// if (fabs(*input_image_center - *input_image_offset) > 20)
						{
							averageA = averageA + *input_image_offset;
							countA++;	
						}
					}
				}
				averageA = averageA / countA;

				double averageB = 0;
				int countB = 0;
				for (int m = -window_size; m <= window_size; m++) {
					for (int n = -window_size; n <= window_size; n++) {
						if (m == 0 && n == 0)
							continue;
						if (b_y+m < 0 || b_y+m >= height || b_x+n < 0 || b_x+n >= width)
							continue;

						int image_offset = (b_y + m) * width + b_x + n;
						Dtype* input_image_offset = input_image + image_offset;
						// if (fabs(*input_image_center - *input_image_offset) > 20)
						{
							averageB = averageB + *input_image_offset;
							countB++;	
						}
					}
				}
				averageB = averageB / countB;

				if (fabs(averageA-averageB) < 50)
					*(output + point_offset) = 1;
			}
		}
	}
}

template<typename Dtype>
void EdgeDetector(cudaStream_t stream, Dtype* input_image, Dtype* input_edge, Dtype* output_preserve, Dtype* output_eliminate, int height, int width, int isSmoothing)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;

	if (isSmoothing == 1){
		// structure extraction for edge-preserving smoothing
		EdgeSelection_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input_edge, output_preserve, height, width);
		EdgeFiltering_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input_edge, output_preserve, height, width);	
	}else{
		// structure extraction for texture removal
		EdgeTexture_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input_image, input_edge, output_eliminate, height, width);		
	}
}