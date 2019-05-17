#include <cuda_runtime.h>
#include <stdio.h>
#include <cudnn.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "layer.h"
using namespace std;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
#define CHECK(call){gpu_error_checker((call),__FILE__,__LINE__);}

inline void gpu_error_checker(cudaError_t error,const char*file, int line)
{
    if (error != cudaSuccess)
    {
        printf("Error:file %s: line %d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
    }
}

double GetTime() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__device__ float relu(float input)
{
    if(input<0)
    {
        input=0;
    }
    return input;
}
__global__ void CONV2DGPU1(float *input_image, float *output_image, float *Kernel,float *bias,
                           int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix;
    if (ix < nx && iy < ny)
    {
        float sum=0;
        for (char r = -1; r < 2; ++r)
        {
            for (char c = -1; c < 2; ++c)
            {
                unsigned int idx = (iy+r)*nx + (ix+c);
                if(!(iy+r==-1 | ix+c==-1 | ix+c==nx | iy+r==ny ))
                {
                    sum+=input_image[idx]*Kernel[3*(r+1)+(c+1)];
                }
            }
        }
        output_image[idx_base]=relu(sum+bias[0]);
    }
}

extern "C" void convolution(float* img_ptr,float** output,int w,int h,layer l)
{

    int image_bytes = w * h * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, img_ptr, image_bytes, cudaMemcpyHostToDevice);

    float* d_output{nullptr};
    int output_size =w * h * sizeof(float);
    cudaMalloc(&d_output, output_size);
    cudaMemset(d_output, 0, output_size);

    int nBytes = l.width*l.height*l.nfilters * sizeof(float);

    float* d_kernel{nullptr};
    cudaMalloc((void**)&d_kernel,nBytes);
    CHECK(cudaMemcpy(d_kernel,l.weight,nBytes, cudaMemcpyHostToDevice));

    float* d_bias{nullptr};
    cudaMalloc((void**)&d_bias,l.nfilters*sizeof(float));
    CHECK(cudaMemcpy(d_bias,l.bias,l.nfilters*sizeof(float),cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    CONV2DGPU1<<<grid,block>>>(d_input,d_output,d_kernel,d_bias,w,h);
    printf("time elapsed:%f \n",GetTime()-t1);
    float* h_output =(float*)malloc(output_size);
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
    //    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bias);
}
