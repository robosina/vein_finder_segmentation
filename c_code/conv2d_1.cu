#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "layer.h"
#include "cudnn.h"
#define DEBUG_MODE 0
#define CPU_DEBUG_MODE 0
#define print 0

#define using_cudnn 1
    using namespace std;

    enum LAYER
{
    CONV2D_1,
    CONV2D_2,
    MAXP2D_1,
    CONV2D_3,
    CONV2D_4,
    MAXP2D_2,
    CONV2D_5,
    CONV2D_6,
    MAXP2D_3,
    CONV2D_7,
    CONV2D_8,
    MAXP2D_4,
    CONV2D_9,
    CONV2D_10,
    UPSM2D_1,
    CONV2D_11,
    CONCAT_1,
    CONV2D_12,
    CONV2D_13,
    UPSM2D_2,
    CONV2D_14,
    CONCAT_2,
    CONV2D_15,
    CONV2D_16,
    UPSM2D_3,
    CONV2D_17,
    CONCAT_3,
    CONV2D_18,
    CONV2D_19,
    UPSM2D_4,
    CONV2D_20,
    CONCAT_4,
    CONV2D_21,
    CONV2D_22,
    CONV2D_23,
    CONV2D_24,


    };

float time_sum=0;
//global variables
//*******conv2d_1**********
float* d_input{0};
float* d_output{0};
float* d_kernel{0};
float* d_bias{0};
//*******conv2d_2**********
float* d_output_2{0};
float* d_kernel_2{0};
float* d_bias_2{0};
//******max_pooling2d_1****
float* d_output_maxp_1{0};
//*******conv2d_3**********
float* d_output_3{0};
float* d_kernel_3{0};
float* d_bias_3{0};
//*******conv2d_4**********
float* d_output_4{0};
float* d_kernel_4{0};
float* d_bias_4{0};
//******max_pooling2d_2****
float* d_output_maxp_2{0};
//*******conv2d_5**********
float* d_output_5{0};
float* d_kernel_5{0};
float* d_bias_5{0};
//*******conv2d_6**********
float* d_output_6{0};
float* d_kernel_6{0};
float* d_bias_6{0};
//******max_pooling2d_3****
float* d_output_maxp_3{0};
//*******conv2d_7**********
float* d_output_7{0};
float* d_kernel_7{0};
float* d_bias_7{0};
//*******conv2d_8**********
float* d_output_8{0};
float* d_kernel_8{0};
float* d_bias_8{0};
//******max_pooling2d_4****
float* d_output_maxp_4{0};
//*******conv2d_9**********
float* d_output_9{0};
float* d_kernel_9{0};
float* d_bias_9{0};
//*******conv2d_10**********
float* d_output_10{0};
float* d_kernel_10{0};
float* d_bias_10{0};
//******max_pooling2d_4****
float* d_output_upsm_1{0};
//*******conv2d_11**********
float* d_output_11{0};
float* d_kernel_11{0};
float* d_bias_11{0};
//******concat_1****
float* d_output_concat_1{0};
//*******conv2d_12**********
float* d_output_12{0};
float* d_kernel_12{0};
float* d_bias_12{0};
//*******conv2d_13**********
float* d_output_13{0};
float* d_kernel_13{0};
float* d_bias_13{0};
//******upsample_2d_2****
float* d_output_upsm_2{0};
//*******conv2d_14**********
float* d_output_14{0};
float* d_kernel_14{0};
float* d_bias_14{0};
//******concat_2****
float* d_output_concat_2{0};
//*******conv2d_15**********
float* d_output_15{0};
float* d_kernel_15{0};
float* d_bias_15{0};
//*******conv2d_16**********
float* d_output_16{0};
float* d_kernel_16{0};
float* d_bias_16{0};
//******upsample_2d_3****
float* d_output_upsm_3{0};
//*******conv2d_17**********
float* d_output_17{0};
float* d_kernel_17{0};
float* d_bias_17{0};
//******concat_3****
float* d_output_concat_3{0};
//*******conv2d_18**********
float* d_output_18{0};
float* d_kernel_18{0};
float* d_bias_18{0};
//*******conv2d_19**********
float* d_output_19{0};
float* d_kernel_19{0};
float* d_bias_19{0};
//******upsample_2d_4****
float* d_output_upsm_4{0};
//*******conv2d_20**********
float* d_output_20{0};
float* d_kernel_20{0};
float* d_bias_20{0};
//******concat_4****
float* d_output_concat_4{0};
//*******conv2d_21**********
float* d_output_21{0};
float* d_kernel_21{0};
float* d_bias_21{0};
//*******conv2d_22**********
float* d_output_22{0};
float* d_kernel_22{0};
float* d_bias_22{0};
//*******conv2d_23**********
float* d_output_23{0};
float* d_kernel_23{0};
float* d_bias_23{0};
//*******conv2d_24**********
float* d_output_24{0};
float* d_kernel_24{0};
float* d_bias_24{0};
float* h_output;

#if using_cudnn==1
//********conv2d_2******************************
cudnnTensorDescriptor_t input_descriptor_2;
cudnnTensorDescriptor_t output_descriptor_2;
cudnnFilterDescriptor_t kernel_descriptor_2;
cudnnConvolutionDescriptor_t convolution_descriptor_2;
cudnnConvolutionFwdAlgo_t convolution_algorithm_2;
size_t workspace_bytes_2 = 0;
void* d_workspace_2{nullptr};
const float alpha = 1, beta = 0;
cudnnHandle_t cudnn_2;

//********conv2d_3******************************
cudnnTensorDescriptor_t input_descriptor_3;
cudnnTensorDescriptor_t output_descriptor_3;
cudnnFilterDescriptor_t kernel_descriptor_3;
cudnnConvolutionDescriptor_t convolution_descriptor_3;
cudnnConvolutionFwdAlgo_t convolution_algorithm_3;
size_t workspace_bytes_3 = 0;
void* d_workspace_3{nullptr};
cudnnHandle_t cudnn_3;

//********conv2d_4******************************
cudnnTensorDescriptor_t input_descriptor_4;
cudnnTensorDescriptor_t output_descriptor_4;
cudnnFilterDescriptor_t kernel_descriptor_4;
cudnnConvolutionDescriptor_t convolution_descriptor_4;
cudnnConvolutionFwdAlgo_t convolution_algorithm_4;
size_t workspace_bytes_4 = 0;
void* d_workspace_4{nullptr};
cudnnHandle_t cudnn_4;
//********conv2d_5******************************
cudnnTensorDescriptor_t input_descriptor_5;
cudnnTensorDescriptor_t output_descriptor_5;
cudnnFilterDescriptor_t kernel_descriptor_5;
cudnnConvolutionDescriptor_t convolution_descriptor_5;
cudnnConvolutionFwdAlgo_t convolution_algorithm_5;
size_t workspace_bytes_5 = 0;
void* d_workspace_5{nullptr};
cudnnHandle_t cudnn_5;
//********conv2d_6******************************
cudnnTensorDescriptor_t input_descriptor_6;
cudnnTensorDescriptor_t output_descriptor_6;
cudnnFilterDescriptor_t kernel_descriptor_6;
cudnnConvolutionDescriptor_t convolution_descriptor_6;
cudnnConvolutionFwdAlgo_t convolution_algorithm_6;
size_t workspace_bytes_6 = 0;
void* d_workspace_6{nullptr};
cudnnHandle_t cudnn_6;
//********conv2d_7******************************
cudnnTensorDescriptor_t input_descriptor_7;
cudnnTensorDescriptor_t output_descriptor_7;
cudnnFilterDescriptor_t kernel_descriptor_7;
cudnnConvolutionDescriptor_t convolution_descriptor_7;
cudnnConvolutionFwdAlgo_t convolution_algorithm_7;
size_t workspace_bytes_7 = 0;
void* d_workspace_7{nullptr};
cudnnHandle_t cudnn_7;
//********conv2d_8******************************
cudnnTensorDescriptor_t input_descriptor_8;
cudnnTensorDescriptor_t output_descriptor_8;
cudnnFilterDescriptor_t kernel_descriptor_8;
cudnnConvolutionDescriptor_t convolution_descriptor_8;
cudnnConvolutionFwdAlgo_t convolution_algorithm_8;
size_t workspace_bytes_8 = 0;
void* d_workspace_8{nullptr};
cudnnHandle_t cudnn_8;
//********conv2d_9******************************
cudnnTensorDescriptor_t input_descriptor_9;
cudnnTensorDescriptor_t output_descriptor_9;
cudnnFilterDescriptor_t kernel_descriptor_9;
cudnnConvolutionDescriptor_t convolution_descriptor_9;
cudnnConvolutionFwdAlgo_t convolution_algorithm_9;
size_t workspace_bytes_9 = 0;
void* d_workspace_9{nullptr};
cudnnHandle_t cudnn_9;
//********conv2d_10******************************
cudnnTensorDescriptor_t input_descriptor_10;
cudnnTensorDescriptor_t output_descriptor_10;
cudnnFilterDescriptor_t kernel_descriptor_10;
cudnnConvolutionDescriptor_t convolution_descriptor_10;
cudnnConvolutionFwdAlgo_t convolution_algorithm_10;
size_t workspace_bytes_10 = 0;
void* d_workspace_10{nullptr};
cudnnHandle_t cudnn_10;
//********conv2d_11******************************
cudnnTensorDescriptor_t input_descriptor_11;
cudnnTensorDescriptor_t output_descriptor_11;
cudnnFilterDescriptor_t kernel_descriptor_11;
cudnnConvolutionDescriptor_t convolution_descriptor_11;
cudnnConvolutionFwdAlgo_t convolution_algorithm_11;
size_t workspace_bytes_11 = 0;
void* d_workspace_11{nullptr};
cudnnHandle_t cudnn_11;
//********conv2d_12******************************
cudnnTensorDescriptor_t input_descriptor_12;
cudnnTensorDescriptor_t output_descriptor_12;
cudnnFilterDescriptor_t kernel_descriptor_12;
cudnnConvolutionDescriptor_t convolution_descriptor_12;
cudnnConvolutionFwdAlgo_t convolution_algorithm_12;
size_t workspace_bytes_12 = 0;
void* d_workspace_12{nullptr};
cudnnHandle_t cudnn_12;
//********conv2d_13******************************
cudnnTensorDescriptor_t input_descriptor_13;
cudnnTensorDescriptor_t output_descriptor_13;
cudnnFilterDescriptor_t kernel_descriptor_13;
cudnnConvolutionDescriptor_t convolution_descriptor_13;
cudnnConvolutionFwdAlgo_t convolution_algorithm_13;
size_t workspace_bytes_13 = 0;
void* d_workspace_13{nullptr};
cudnnHandle_t cudnn_13;
//********conv2d_14******************************
cudnnTensorDescriptor_t input_descriptor_14;
cudnnTensorDescriptor_t output_descriptor_14;
cudnnFilterDescriptor_t kernel_descriptor_14;
cudnnConvolutionDescriptor_t convolution_descriptor_14;
cudnnConvolutionFwdAlgo_t convolution_algorithm_14;
size_t workspace_bytes_14 = 0;
void* d_workspace_14{nullptr};
cudnnHandle_t cudnn_14;
//********conv2d_15******************************
cudnnTensorDescriptor_t input_descriptor_15;
cudnnTensorDescriptor_t output_descriptor_15;
cudnnFilterDescriptor_t kernel_descriptor_15;
cudnnConvolutionDescriptor_t convolution_descriptor_15;
cudnnConvolutionFwdAlgo_t convolution_algorithm_15;
size_t workspace_bytes_15 = 0;
void* d_workspace_15{nullptr};
cudnnHandle_t cudnn_15;
//********conv2d_16******************************
cudnnTensorDescriptor_t input_descriptor_16;
cudnnTensorDescriptor_t output_descriptor_16;
cudnnFilterDescriptor_t kernel_descriptor_16;
cudnnConvolutionDescriptor_t convolution_descriptor_16;
cudnnConvolutionFwdAlgo_t convolution_algorithm_16;
size_t workspace_bytes_16 = 0;
void* d_workspace_16{nullptr};
cudnnHandle_t cudnn_16;

//********conv2d_17******************************
cudnnTensorDescriptor_t input_descriptor_17;
cudnnTensorDescriptor_t output_descriptor_17;
cudnnFilterDescriptor_t kernel_descriptor_17;
cudnnConvolutionDescriptor_t convolution_descriptor_17;
cudnnConvolutionFwdAlgo_t convolution_algorithm_17;
size_t workspace_bytes_17 = 0;
void* d_workspace_17{nullptr};
cudnnHandle_t cudnn_17;
//********conv2d_18******************************
cudnnTensorDescriptor_t input_descriptor_18;
cudnnTensorDescriptor_t output_descriptor_18;
cudnnFilterDescriptor_t kernel_descriptor_18;
cudnnConvolutionDescriptor_t convolution_descriptor_18;
cudnnConvolutionFwdAlgo_t convolution_algorithm_18;
size_t workspace_bytes_18 = 0;
void* d_workspace_18{nullptr};
cudnnHandle_t cudnn_18;
//********conv2d_19******************************
cudnnTensorDescriptor_t input_descriptor_19;
cudnnTensorDescriptor_t output_descriptor_19;
cudnnFilterDescriptor_t kernel_descriptor_19;
cudnnConvolutionDescriptor_t convolution_descriptor_19;
cudnnConvolutionFwdAlgo_t convolution_algorithm_19;
size_t workspace_bytes_19 = 0;
void* d_workspace_19{nullptr};
cudnnHandle_t cudnn_19;
//********conv2d_20******************************
cudnnTensorDescriptor_t input_descriptor_20;
cudnnTensorDescriptor_t output_descriptor_20;
cudnnFilterDescriptor_t kernel_descriptor_20;
cudnnConvolutionDescriptor_t convolution_descriptor_20;
cudnnConvolutionFwdAlgo_t convolution_algorithm_20;
size_t workspace_bytes_20 = 0;
void* d_workspace_20{nullptr};
cudnnHandle_t cudnn_20;
//********conv2d_21******************************
cudnnTensorDescriptor_t input_descriptor_21;
cudnnTensorDescriptor_t output_descriptor_21;
cudnnFilterDescriptor_t kernel_descriptor_21;
cudnnConvolutionDescriptor_t convolution_descriptor_21;
cudnnConvolutionFwdAlgo_t convolution_algorithm_21;
size_t workspace_bytes_21 = 0;
void* d_workspace_21{nullptr};
cudnnHandle_t cudnn_21;
//********conv2d_22******************************
cudnnTensorDescriptor_t input_descriptor_22;
cudnnTensorDescriptor_t output_descriptor_22;
cudnnFilterDescriptor_t kernel_descriptor_22;
cudnnConvolutionDescriptor_t convolution_descriptor_22;
cudnnConvolutionFwdAlgo_t convolution_algorithm_22;
size_t workspace_bytes_22 = 0;
void* d_workspace_22{nullptr};
cudnnHandle_t cudnn_22;
//********conv2d_23******************************
cudnnTensorDescriptor_t input_descriptor_23;
cudnnTensorDescriptor_t output_descriptor_23;
cudnnFilterDescriptor_t kernel_descriptor_23;
cudnnConvolutionDescriptor_t convolution_descriptor_23;
cudnnConvolutionFwdAlgo_t convolution_algorithm_23;
size_t workspace_bytes_23 = 0;
void* d_workspace_23{nullptr};
cudnnHandle_t cudnn_23;

#endif

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

__device__ float sigmoid(float input)
{
    return 1/(1+exp(-input));
}
__global__ void CONV2DGPU1(float *input_image, float *output_image, float *Kernel,float *bias,
                           int nx, int ny,int layer) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix;
    if (ix < nx && iy < ny)
    {
        float sum=0;
        int which_layer=layer*9;
        if(iy!=0)
        {
            if(ix!=0)
            {
                unsigned int idx11 = (iy-1)*nx + (ix-1);
                sum+=input_image[idx11]*Kernel[which_layer];
            }

            unsigned int idx12 = (iy-1)*nx + (ix);
            sum+=input_image[idx12]*Kernel[1+which_layer];
            if(ix!=nx-1)
            {
                unsigned int idx13 = (iy-1)*nx + (ix+1);
                sum+=input_image[idx13]*Kernel[2+which_layer];
            }
        }

        if(ix!=0)
        {
            unsigned int idx21 = (iy)*nx + (ix-1);
            sum+=input_image[idx21]*Kernel[3+which_layer];
        }

        unsigned int idx22 = (iy)*nx + (ix);
        sum+=input_image[idx22]*Kernel[4+which_layer];

        if(ix!=nx-1)
        {
            unsigned int idx23 = (iy)*nx + (ix+1);
            sum+=input_image[idx23]*Kernel[5+which_layer];
        }
        if(iy!=ny-1)
        {
            if(ix!=0)
            {
                unsigned int idx31 = (iy+1)*nx + (ix-1);
                sum+=input_image[idx31]*Kernel[6+which_layer];
            }

            if(iy!=ny-1)
            {
                unsigned int idx32 = (iy+1)*nx + (ix);
                sum+=input_image[idx32]*Kernel[7+which_layer];
            }
            if(ix!=nx-1 && iy!=ny-1)
            {
                unsigned int idx33 = (iy+1)*nx + (ix+1);
                sum+=input_image[idx33]*Kernel[8+which_layer];
            }
        }
        output_image[idx_base+nx*ny*layer]=relu(sum+bias[layer]);
    }
}
__global__ void  CONV2D_2_GPU1(float *input_image, float *output_image, float *Kernel,float *bias,
                              int nx, int ny,int layer,int depth)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix;
    if (ix < nx && iy < ny)
    {
        float sum=0;
#pragma unroll 1
        for (int dth = 0; dth < depth; ++dth)    //dth---> depth
        {
            int depth_index=dth*nx*ny;
            for (char r = -1; r < 2; ++r)
            {
                for (char c = -1; c < 2; ++c)
                {
                    unsigned int idx = (iy+r)*nx + (ix+c) + depth_index;
                    if(!(iy+r==-1 | ix+c==-1 | ix+c==nx | iy+r==ny ))
                    {
                        sum+=input_image[idx]*Kernel[3*(r+1)+(c+1)+dth*9+layer*9*depth];
                    }
                }
            }
        }
        output_image[idx_base+nx*ny*layer]=relu(sum+bias[layer]);
    }
}

__global__ void CONV2D_2_GPU1X1(float *input_image, float *output_image, float *Kernel,float *bias,
                                int nx, int ny,int layer,int depth)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix;
    if (ix < nx && iy < ny)
    {
        float sum=0;
#pragma unroll 1
        for (int dth = 0; dth < depth; ++dth)    //dth---> depth
        {
            int depth_index=dth*nx*ny;
            unsigned int idx = iy*nx+ix+depth_index;
            sum+=input_image[idx]*Kernel[dth];
        }
        output_image[idx_base]=sigmoid(sum+bias[layer]);
    }
}
__global__ void add_bias(float *output_image,float *bias,int nx, int ny,int layer_size)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny)
    {
        for (int layer = 0; layer < layer_size; ++layer)
        {
            unsigned int idx_base = iy*nx + ix + layer*nx*ny;
            output_image[idx_base]=relu(output_image[idx_base]+bias[layer]);

        }
    }
}

__global__ void MAXP2D_GPU(float *input_image, float *output_image,int nx, int ny,int layer)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix + layer*nx*ny;
    if (ix < nx && iy < ny)
    {
        unsigned int x_index=2*ix;
        unsigned int offset=layer*2*nx*2*ny;

        unsigned int idx_1= (x_index)+(   (2*iy)*(2*nx)  )+offset;  //top-left
        unsigned int idx_2= (x_index+1)+(   (2*iy)*(2*nx)  )+offset;  //top-right
        unsigned int idx_3= (x_index)+(   (2*iy+1)*(2*nx)  )+offset;  //bottom-left
        unsigned int idx_4= (x_index+1)+(   (2*iy+1)*(2*nx)  )+offset;  //bottom-right
        float max;
        if(input_image[idx_1]>input_image[idx_2])
        {
            max=input_image[idx_1];
        }
        else
        {
            max=input_image[idx_2];
        }

        if(max<input_image[idx_3])
        {
            max=input_image[idx_3];
        }

        if(max<input_image[idx_4])
        {
            max=input_image[idx_4];
        }
        output_image[idx_base]=max;
        //        printf("idx:%d = %f\n",idx_base,output_image[idx_base]);
        //        printf("ix:%d iy:%d index number:%d tl:%f tr:%f bl:%f br:%f output:%f \n",ix,iy,idx_base,
        //               input_image[idx_1],input_image[idx_2],input_image[idx_3],input_image[idx_4],output_image[idx_base]);
    }
}

__global__ void UPSM_2D_GPU(float *input_image, float *output_image,int nx, int ny,int layer)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix + layer*nx*ny;
    if (ix < nx && iy < ny)
    {
        unsigned int x_index=2*ix;
        unsigned int offset=layer*2*nx*2*ny;

        unsigned int idx_1= (x_index)+(   (2*iy)*(2*nx)  )+offset;  //top-left
        unsigned int idx_2= (x_index+1)+(   (2*iy)*(2*nx)  )+offset;  //top-right
        unsigned int idx_3= (x_index)+(   (2*iy+1)*(2*nx)  )+offset;  //bottom-left
        unsigned int idx_4= (x_index+1)+(   (2*iy+1)*(2*nx)  )+offset;  //bottom-right
        output_image[idx_1]=input_image[idx_base];
        output_image[idx_2]=input_image[idx_base];
        output_image[idx_3]=input_image[idx_base];
        output_image[idx_4]=input_image[idx_base];
    }
}

__global__ void CONCAT_GPU(float *input_image1,  //first volume
                           float *input_image2,  //second volume
                           float *output_image,  //concat first volume and second volume
                           int nx,
                           int ny,
                           int layer,            //which layer is under process
                           int NLayer )          //number of total layers
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base1 = iy*nx + ix + layer*nx*ny;
    unsigned int idx_base2 = iy*nx + ix + (NLayer/2+layer)*nx*ny;
    if (ix < nx && iy < ny)
    {
        output_image[idx_base1]=input_image1[idx_base1];
        output_image[idx_base2]=input_image2[idx_base1];
        //        printf("ix:%d, iy:%d, index:%d, pixel1:%f,pixel2:%f\n",ix,iy,idx_base1,input_image1[idx_base1],input_image2[idx_base1]);
    }
}

extern "C" void conv2d_1(float* img_ptr,int w,int h,layer l)
{
    time_sum=0;
    cudaMemcpy(d_input, img_ptr, l.input_size, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)
    {
        CONV2DGPU1<<<grid,block>>>(d_input,d_output,d_kernel,d_bias,w,h,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_1:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif

}

extern "C" void conv2d_2(int w, int h, layer l)
{

#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_2,
                                       &alpha,
                                       input_descriptor_2,
                                       d_output,
                                       kernel_descriptor_2,
                                       d_kernel_2,
                                       convolution_descriptor_2,
                                       convolution_algorithm_2,
                                       d_workspace_2,
                                       workspace_bytes_2,
                                       &beta,
                                       output_descriptor_2,
                                       d_output_2));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_2,d_bias_2,256,256,16);
#if print==1
    printf("time elapsed \033[1;35mconv2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output,d_output_2,d_kernel_2,d_bias_2,w,h,i,l.depth);
    }
#if print==1
    printf("time elapsed \033[1;35mconv2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
    cudaDeviceSynchronize();
#endif
}

extern "C" void maxp2d_1(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_2,d_output_maxp_1,w,h,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mmaxp2d_1:%f msec \n\033[0m",1000*(GetTime()-t1));
#endif
}
extern "C" void conv2d_3(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_3,
                                       &alpha,
                                       input_descriptor_3,
                                       d_output_maxp_1,
                                       kernel_descriptor_3,
                                       d_kernel_3,
                                       convolution_descriptor_3,
                                       convolution_algorithm_3,
                                       d_workspace_3,
                                       workspace_bytes_3,
                                       &beta,
                                       output_descriptor_3,
                                       d_output_3));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_3,d_bias_3,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_3:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_maxp_1,d_output_3,d_kernel_3,d_bias_3,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_3:%f msec \n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_4(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_4,
                                       &alpha,
                                       input_descriptor_4,
                                       d_output_3,
                                       kernel_descriptor_4,
                                       d_kernel_4,
                                       convolution_descriptor_4,
                                       convolution_algorithm_4,
                                       d_workspace_4,
                                       workspace_bytes_4,
                                       &beta,
                                       output_descriptor_4,
                                       d_output_4));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_4,d_bias_4,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_4:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_3,d_output_4,d_kernel_4,d_bias_4,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_4:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void maxp2d_2(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_4,d_output_maxp_2,w,h,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mmaxp2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}
extern "C" void conv2d_5(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_5,
                                       &alpha,
                                       input_descriptor_5,
                                       d_output_maxp_2,
                                       kernel_descriptor_5,
                                       d_kernel_5,
                                       convolution_descriptor_5,
                                       convolution_algorithm_5,
                                       d_workspace_5,
                                       workspace_bytes_5,
                                       &beta,
                                       output_descriptor_5,
                                       d_output_5));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_5,d_bias_5,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_5:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_maxp_2,d_output_5,d_kernel_5,d_bias_5,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_5:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_6(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_6,
                                       &alpha,
                                       input_descriptor_6,
                                       d_output_5,
                                       kernel_descriptor_6,
                                       d_kernel_6,
                                       convolution_descriptor_6,
                                       convolution_algorithm_6,
                                       d_workspace_6,
                                       workspace_bytes_6,
                                       &beta,
                                       output_descriptor_6,
                                       d_output_6));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_6,d_bias_6,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_6:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_5,d_output_6,d_kernel_6,d_bias_6,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_6:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void maxp2d_3(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_6,d_output_maxp_3,w,h,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mmaxp2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif

}

extern "C" void conv2d_7(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_7,
                                       &alpha,
                                       input_descriptor_7,
                                       d_output_maxp_3,
                                       kernel_descriptor_7,
                                       d_kernel_7,
                                       convolution_descriptor_7,
                                       convolution_algorithm_7,
                                       d_workspace_7,
                                       workspace_bytes_7,
                                       &beta,
                                       output_descriptor_7,
                                       d_output_7));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_7,d_bias_7,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_7:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_maxp_3,d_output_7,d_kernel_7,d_bias_7,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_7:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_8(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_8,
                                       &alpha,
                                       input_descriptor_8,
                                       d_output_7,
                                       kernel_descriptor_8,
                                       d_kernel_8,
                                       convolution_descriptor_8,
                                       convolution_algorithm_8,
                                       d_workspace_8,
                                       workspace_bytes_8,
                                       &beta,
                                       output_descriptor_8,
                                       d_output_8));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_8,d_bias_8,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_8:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_7,d_output_8,d_kernel_8,d_bias_8,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_8:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void maxp2d_4(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_8,d_output_maxp_4,w,h,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mmaxp2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}

extern "C" void conv2d_9(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_9,
                                       &alpha,
                                       input_descriptor_9,
                                       d_output_maxp_4,
                                       kernel_descriptor_9,
                                       d_kernel_9,
                                       convolution_descriptor_9,
                                       convolution_algorithm_9,
                                       d_workspace_9,
                                       workspace_bytes_9,
                                       &beta,
                                       output_descriptor_9,
                                       d_output_9));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_9,d_bias_9,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_9:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_maxp_4,d_output_9,d_kernel_9,d_bias_9,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_9:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_10(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_10,
                                       &alpha,
                                       input_descriptor_10,
                                       d_output_9,
                                       kernel_descriptor_10,
                                       d_kernel_10,
                                       convolution_descriptor_10,
                                       convolution_algorithm_10,
                                       d_workspace_10,
                                       workspace_bytes_10,
                                       &beta,
                                       output_descriptor_10,
                                       d_output_10));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    add_bias<<<grid,block>>>(d_output_10,d_bias_10,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_10:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_9,d_output_10,d_kernel_10,d_bias_10,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_10:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void upsample_2d_1(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w/2 + block.x - 1) / block.x, (h/2 + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        UPSM_2D_GPU<<<grid,block>>>(d_output_10,d_output_upsm_1,w/2,h/2,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_1:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}

extern "C" void conv2d_11(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_11,
                                       &alpha,
                                       input_descriptor_11,
                                       d_output_upsm_1,
                                       kernel_descriptor_11,
                                       d_kernel_11,
                                       convolution_descriptor_11,
                                       convolution_algorithm_11,
                                       d_workspace_11,
                                       workspace_bytes_11,
                                       &beta,
                                       output_descriptor_11,
                                       d_output_11));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_11,d_bias_11,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_11:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_upsm_1,d_output_11,d_kernel_11,d_bias_11,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_11:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void concat_1(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters/2; ++i)   //which volume is selected for conv
    {
        CONCAT_GPU<<<grid,block>>>(d_output_8,d_output_11,d_output_concat_1,w,h,i,l.nfilters);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_1:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}

extern "C" void conv2d_12(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_12,
                                       &alpha,
                                       input_descriptor_12,
                                       d_output_concat_1,
                                       kernel_descriptor_12,
                                       d_kernel_12,
                                       convolution_descriptor_12,
                                       convolution_algorithm_12,
                                       d_workspace_12,
                                       workspace_bytes_12,
                                       &beta,
                                       output_descriptor_12,
                                       d_output_12));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_12,d_bias_12,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_12:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_concat_1,d_output_12,d_kernel_12,d_bias_12,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_12:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_13(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_13,
                                       &alpha,
                                       input_descriptor_13,
                                       d_output_12,
                                       kernel_descriptor_13,
                                       d_kernel_13,
                                       convolution_descriptor_13,
                                       convolution_algorithm_13,
                                       d_workspace_13,
                                       workspace_bytes_13,
                                       &beta,
                                       output_descriptor_13,
                                       d_output_13));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_13,d_bias_13,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_13:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_12,d_output_13,d_kernel_13,d_bias_13,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_13:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void upsample_2d_2(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w/2 + block.x - 1) / block.x, (h/2 + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        UPSM_2D_GPU<<<grid,block>>>(d_output_13,d_output_upsm_2,w/2,h/2,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}

extern "C" void conv2d_14(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_14,
                                       &alpha,
                                       input_descriptor_14,
                                       d_output_upsm_2,
                                       kernel_descriptor_14,
                                       d_kernel_14,
                                       convolution_descriptor_14,
                                       convolution_algorithm_14,
                                       d_workspace_14,
                                       workspace_bytes_14,
                                       &beta,
                                       output_descriptor_14,
                                       d_output_14));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_14,d_bias_14,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_14:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_upsm_2,d_output_14,d_kernel_14,d_bias_14,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_14:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void concat_2(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters/2; ++i)   //which volume is selected for conv
    {
        CONCAT_GPU<<<grid,block>>>(d_output_6,d_output_14,d_output_concat_2,w,h,i,l.nfilters);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_1:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}

extern "C" void conv2d_15(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_15,
                                       &alpha,
                                       input_descriptor_15,
                                       d_output_concat_2,
                                       kernel_descriptor_15,
                                       d_kernel_15,
                                       convolution_descriptor_15,
                                       convolution_algorithm_15,
                                       d_workspace_15,
                                       workspace_bytes_15,
                                       &beta,
                                       output_descriptor_15,
                                       d_output_15));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_15,d_bias_15,w,h,l.nfilters);
#if print==1
    printf("time elapsed \033[1;35mconv2d_15:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_concat_2,d_output_15,d_kernel_15,d_bias_15,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_15:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_16(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_16,
                                       &alpha,
                                       input_descriptor_16,
                                       d_output_15,
                                       kernel_descriptor_16,
                                       d_kernel_16,
                                       convolution_descriptor_16,
                                       convolution_algorithm_16,
                                       d_workspace_16,
                                       workspace_bytes_16,
                                       &beta,
                                       output_descriptor_16,
                                       d_output_16));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_16,d_bias_16,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_16:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_15,d_output_16,d_kernel_16,d_bias_16,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_16:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}
extern "C" void upsample_2d_3(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w/2 + block.x - 1) / block.x, (h/2 + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        UPSM_2D_GPU<<<grid,block>>>(d_output_16,d_output_upsm_3,w/2,h/2,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}
extern "C" void conv2d_17(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_17,
                                       &alpha,
                                       input_descriptor_17,
                                       d_output_upsm_3,
                                       kernel_descriptor_17,
                                       d_kernel_17,
                                       convolution_descriptor_17,
                                       convolution_algorithm_17,
                                       d_workspace_17,
                                       workspace_bytes_17,
                                       &beta,
                                       output_descriptor_17,
                                       d_output_17));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_17,d_bias_17,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_17:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_upsm_3,d_output_17,d_kernel_17,d_bias_17,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_17:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void concat_3(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters/2; ++i)   //which volume is selected for conv
    {
        CONCAT_GPU<<<grid,block>>>(d_output_4,d_output_17,d_output_concat_3,w,h,i,l.nfilters);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_1:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}
extern "C" void conv2d_18(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_18,
                                       &alpha,
                                       input_descriptor_18,
                                       d_output_concat_3,
                                       kernel_descriptor_18,
                                       d_kernel_18,
                                       convolution_descriptor_18,
                                       convolution_algorithm_18,
                                       d_workspace_18,
                                       workspace_bytes_18,
                                       &beta,
                                       output_descriptor_18,
                                       d_output_18));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_18,d_bias_18,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_18:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_concat_3,d_output_18,d_kernel_18,d_bias_18,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_18:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}
extern "C" void conv2d_19(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_19,
                                       &alpha,
                                       input_descriptor_19,
                                       d_output_18,
                                       kernel_descriptor_19,
                                       d_kernel_19,
                                       convolution_descriptor_19,
                                       convolution_algorithm_19,
                                       d_workspace_19,
                                       workspace_bytes_19,
                                       &beta,
                                       output_descriptor_19,
                                       d_output_19));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_19,d_bias_19,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_19:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_18,d_output_19,d_kernel_19,d_bias_19,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_19:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}
extern "C" void upsample_2d_4(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w/2 + block.x - 1) / block.x, (h/2 + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        UPSM_2D_GPU<<<grid,block>>>(d_output_19,d_output_upsm_4,w/2,h/2,i);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_2:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}
extern "C" void conv2d_20(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_20,
                                       &alpha,
                                       input_descriptor_20,
                                       d_output_upsm_4,
                                       kernel_descriptor_20,
                                       d_kernel_20,
                                       convolution_descriptor_20,
                                       convolution_algorithm_20,
                                       d_workspace_20,
                                       workspace_bytes_20,
                                       &beta,
                                       output_descriptor_20,
                                       d_output_20));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_20,d_bias_20,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_20:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_upsm_4,d_output_20,d_kernel_20,d_bias_20,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_20:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void concat_4(int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters/2; ++i)   //which volume is selected for conv
    {
        CONCAT_GPU<<<grid,block>>>(d_output_2,d_output_20,d_output_concat_4,w,h,i,l.nfilters);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mupsample_1:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
}
extern "C" void conv2d_21(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_21,
                                       &alpha,
                                       input_descriptor_21,
                                       d_output_concat_4,
                                       kernel_descriptor_21,
                                       d_kernel_21,
                                       convolution_descriptor_21,
                                       convolution_algorithm_21,
                                       d_workspace_21,
                                       workspace_bytes_21,
                                       &beta,
                                       output_descriptor_21,
                                       d_output_21));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_21,d_bias_21,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_21:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_concat_4,d_output_21,d_kernel_21,d_bias_21,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;33mconv2d_21:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}
extern "C" void conv2d_22(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_22,
                                       &alpha,
                                       input_descriptor_22,
                                       d_output_21,
                                       kernel_descriptor_22,
                                       d_kernel_22,
                                       convolution_descriptor_22,
                                       convolution_algorithm_22,
                                       d_workspace_22,
                                       workspace_bytes_22,
                                       &beta,
                                       output_descriptor_22,
                                       d_output_22));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_22,d_bias_22,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_22:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_21,d_output_22,d_kernel_22,d_bias_22,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;34mconv2d_22:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}
extern "C" void conv2d_23(int w, int h, layer l)
{
#if using_cudnn==1
    double t1=GetTime();
    checkCUDNN(cudnnConvolutionForward(cudnn_23,
                                       &alpha,
                                       input_descriptor_23,
                                       d_output_22,
                                       kernel_descriptor_23,
                                       d_kernel_23,
                                       convolution_descriptor_23,
                                       convolution_algorithm_23,
                                       d_workspace_23,
                                       workspace_bytes_23,
                                       &beta,
                                       output_descriptor_23,
                                       d_output_23));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    add_bias<<<grid,block>>>(d_output_23,d_bias_23,w,h,l.nfilters);
    printf("time elapsed \033[1;35mconv2d_23:%f msec\n\033[0m",1000*(GetTime()-t1));
#else
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();

    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output_22,d_output_23,d_kernel_23,d_bias_23,
                                       w,h,i,l.depth);
    }
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;34mconv2d_23:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
#endif
}

extern "C" void conv2d_24(float** output, int w, int h, layer l)
{
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();

    CONV2D_2_GPU1X1<<<grid,block>>>(d_output_23,d_output_24,d_kernel_24,d_bias_24,
                                     w,h,0,l.depth);
    cudaDeviceSynchronize();
#if print==1
    printf("time elapsed \033[1;34mconv2d_24:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
    cudaMemcpy(h_output, d_output_24,l.output_size, cudaMemcpyDeviceToHost);
#if print==1
    printf("time elapsed \033[1;35mCopyTime:%f msec\n\033[0m",1000*(GetTime()-t1));
#endif
    *output=h_output;
}
extern "C" void LOAD_NEURAL_NETWORK(LAYER Layer, int w, int h, layer& l)
{
    switch (Layer) {
    case CONV2D_1:
    {
        l.input_size = w * h * sizeof(float);
        cudaMalloc(&d_input, l.input_size);

        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_kernel,l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias,l.bias_size);

        cudaMemset(d_output, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_1: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_2:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_2, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_2, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_2, l.bias_size);

        cudaMemset(d_output_2, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_2,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_2,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_2: image:%d,%d \n\033[0m",w,h);

#if using_cudnn==1
        cudnnCreate(&cudnn_2);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_2));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_2,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/l.im_h,
                                              /*image_width=*/l.im_w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_2));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_2,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/l.im_h,
                                              /*image_width=*/l.im_w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_2));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_2,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/16,
                                              /*in_channels=*/16,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_2));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_2,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_2,
                                                input_descriptor_2,
                                                kernel_descriptor_2,
                                                convolution_descriptor_2,
                                                output_descriptor_2,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_2));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_2,
                                                           input_descriptor_2,
                                                           kernel_descriptor_2,
                                                           convolution_descriptor_2,
                                                           output_descriptor_2,
                                                           convolution_algorithm_2,
                                                           &workspace_bytes_2));


        cudaMalloc(&d_workspace_2, workspace_bytes_2);



#endif
        break;
    }
    case MAXP2D_1:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_1, l.output_size);

        l.im_h=h;
        l.im_w=w;

        cudaMemset(d_output_maxp_1, 0, l.output_size);

        printf("\033[1;31mLOAD MAXP2D_1: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_3:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_3, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_3, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_3, l.bias_size);

        cudaMemset(d_output_3, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_3,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_3,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_3: image:%d,%d \n\033[0m",w,h);

#if using_cudnn==1
        //   128x128x16   ----> 128x128x32
        cudnnCreate(&cudnn_3);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_3));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_3,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_3));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_3,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_3));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_3,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/32,
                                              /*in_channels=*/16,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_3));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_3,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_3,
                                                input_descriptor_3,
                                                kernel_descriptor_3,
                                                convolution_descriptor_3,
                                                output_descriptor_3,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_3));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_3,
                                                           input_descriptor_3,
                                                           kernel_descriptor_3,
                                                           convolution_descriptor_3,
                                                           output_descriptor_3,
                                                           convolution_algorithm_3,
                                                           &workspace_bytes_3));


        cudaMalloc(&d_workspace_3, workspace_bytes_3);



#endif
        break;
    }
    case CONV2D_4:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_4, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_4, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_4, l.bias_size);

        cudaMemset(d_output_4, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_4,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_4,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_4: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   128x128x32   ----> 128x128x32
        cudnnCreate(&cudnn_4);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_4));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_4,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_4));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_4,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_4));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_4,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/32,
                                              /*in_channels=*/32,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_4));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_4,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_4,
                                                input_descriptor_4,
                                                kernel_descriptor_4,
                                                convolution_descriptor_4,
                                                output_descriptor_4,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_4));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_4,
                                                           input_descriptor_4,
                                                           kernel_descriptor_4,
                                                           convolution_descriptor_4,
                                                           output_descriptor_4,
                                                           convolution_algorithm_4,
                                                           &workspace_bytes_4));


        cudaMalloc(&d_workspace_4, workspace_bytes_4);



#endif
        break;
    }
    case MAXP2D_2:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_2, l.output_size);

        l.im_h=h;
        l.im_w=w;

        cudaMemset(d_output_maxp_2, 0, l.output_size);

        printf("\033[1;31mLOAD MAXP2D_2: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_5:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_5, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_5, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_5, l.bias_size);

        cudaMemset(d_output_5, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_5,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_5,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_5: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   64x64x32 ----> 64x64x64
        cudnnCreate(&cudnn_5);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_5));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_5,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_5));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_5,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_5));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_5,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/32,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_5));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_5,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_5,
                                                input_descriptor_5,
                                                kernel_descriptor_5,
                                                convolution_descriptor_5,
                                                output_descriptor_5,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_5));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_5,
                                                           input_descriptor_5,
                                                           kernel_descriptor_5,
                                                           convolution_descriptor_5,
                                                           output_descriptor_5,
                                                           convolution_algorithm_5,
                                                           &workspace_bytes_5));


        cudaMalloc(&d_workspace_5, workspace_bytes_5);



#endif

        break;
    }
    case CONV2D_6:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_6, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_6, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_6, l.bias_size);

        cudaMemset(d_output_6, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_6,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_6,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_6: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   64x64x64   ----> 64x64x64
        cudnnCreate(&cudnn_6);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_6));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_6,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_6));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_6,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_6));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_6,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_6));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_6,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_6,
                                                input_descriptor_6,
                                                kernel_descriptor_6,
                                                convolution_descriptor_6,
                                                output_descriptor_6,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_6));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_6,
                                                           input_descriptor_6,
                                                           kernel_descriptor_6,
                                                           convolution_descriptor_6,
                                                           output_descriptor_6,
                                                           convolution_algorithm_6,
                                                           &workspace_bytes_6));


        cudaMalloc(&d_workspace_6, workspace_bytes_6);



#endif
        break;
    }
    case MAXP2D_3:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_3, l.output_size);

        l.im_h=h;
        l.im_w=w;

        cudaMemset(d_output_maxp_3, 0, l.output_size);

        printf("\033[1;31mLOAD MAXP2D_3: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_7:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_7, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_7, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_7, l.bias_size);

        cudaMemset(d_output_7, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_7,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_7,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_7: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   64x64x64   ----> 32x32x64
        cudnnCreate(&cudnn_7);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_7));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_7,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_7));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_7,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_7));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_7,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_7));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_7,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_7,
                                                input_descriptor_7,
                                                kernel_descriptor_7,
                                                convolution_descriptor_7,
                                                output_descriptor_7,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_7));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_7,
                                                           input_descriptor_7,
                                                           kernel_descriptor_7,
                                                           convolution_descriptor_7,
                                                           output_descriptor_7,
                                                           convolution_algorithm_7,
                                                           &workspace_bytes_7));


        cudaMalloc(&d_workspace_7, workspace_bytes_7);
#endif
        break;
    }
    case CONV2D_8:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_8, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_8, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_8, l.bias_size);

        cudaMemset(d_output_8, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_8,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_8,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_8: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   32x32x64   ----> 32x32x64
        cudnnCreate(&cudnn_8);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_8));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_8,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_8));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_8,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_8));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_8,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_8));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_8,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_8,
                                                input_descriptor_8,
                                                kernel_descriptor_8,
                                                convolution_descriptor_8,
                                                output_descriptor_8,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_8));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_8,
                                                           input_descriptor_8,
                                                           kernel_descriptor_8,
                                                           convolution_descriptor_8,
                                                           output_descriptor_8,
                                                           convolution_algorithm_8,
                                                           &workspace_bytes_8));


        cudaMalloc(&d_workspace_8, workspace_bytes_8);



#endif
        break;
    }
    case MAXP2D_4:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_4, l.output_size);

        cudaMemset(d_output_maxp_4, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD MAXP2D_4: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_9:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_9, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_9, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_9, l.bias_size);

        cudaMemset(d_output_9, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_9,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_9,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_9: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   16x16x64   ----> 16x16x128
        cudnnCreate(&cudnn_9);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_9));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_9,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_9));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_9,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/128,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_9));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_9,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/128,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_9));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_9,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_9,
                                                input_descriptor_9,
                                                kernel_descriptor_9,
                                                convolution_descriptor_9,
                                                output_descriptor_9,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_9));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_9,
                                                           input_descriptor_9,
                                                           kernel_descriptor_9,
                                                           convolution_descriptor_9,
                                                           output_descriptor_9,
                                                           convolution_algorithm_9,
                                                           &workspace_bytes_9));


        cudaMalloc(&d_workspace_9, workspace_bytes_9);



#endif
        break;
    }
    case CONV2D_10:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_10, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_10, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_10, l.bias_size);

        cudaMemset(d_output_10, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_10,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_10,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_10: image:%d,%d \n\033[0m",w,h);

#if using_cudnn==1
        //   16x16x128   ----> 16x16x128
        cudnnCreate(&cudnn_10);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_10));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_10,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/128,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_10));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_10,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/128,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_10));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_10,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/128,
                                              /*in_channels=*/128,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_10));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_10,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_10,
                                                input_descriptor_10,
                                                kernel_descriptor_10,
                                                convolution_descriptor_10,
                                                output_descriptor_10,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_10));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_10,
                                                           input_descriptor_10,
                                                           kernel_descriptor_10,
                                                           convolution_descriptor_10,
                                                           output_descriptor_10,
                                                           convolution_algorithm_10,
                                                           &workspace_bytes_10));


        cudaMalloc(&d_workspace_10, workspace_bytes_10);



#endif
        break;
    }
    case UPSM2D_1:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_upsm_1, l.output_size);

        cudaMemset(d_output_upsm_1, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD UPSM2D_1: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_11:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_11, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_11, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_11, l.bias_size);

        cudaMemset(d_output_11, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_11,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_11,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_11: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   32x32x128   ----> 32x32x64
        cudnnCreate(&cudnn_11);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_11));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_11,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/128,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_11));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_11,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_11));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_11,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/128,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_11));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_11,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_11,
                                                input_descriptor_11,
                                                kernel_descriptor_11,
                                                convolution_descriptor_11,
                                                output_descriptor_11,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_11));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_11,
                                                           input_descriptor_11,
                                                           kernel_descriptor_11,
                                                           convolution_descriptor_11,
                                                           output_descriptor_11,
                                                           convolution_algorithm_11,
                                                           &workspace_bytes_11));


        cudaMalloc(&d_workspace_11, workspace_bytes_11);



#endif
        break;
    }
    case CONCAT_1:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_concat_1, l.output_size);

        cudaMemset(d_output_concat_1, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD CONCAT_1: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_12:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_12, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_12, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_12, l.bias_size);

        l.im_h=h;
        l.im_w=w;

        cudaMemset(d_output_12, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_12,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_12,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        printf("\033[1;31mLOAD CONV2D_12: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   32x32x128   ----> 32x32x64
        cudnnCreate(&cudnn_12);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_12));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_12,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/128,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_12));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_12,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_12));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_12,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/128,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_12));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_12,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_12,
                                                input_descriptor_12,
                                                kernel_descriptor_12,
                                                convolution_descriptor_12,
                                                output_descriptor_12,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_12));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_12,
                                                           input_descriptor_12,
                                                           kernel_descriptor_12,
                                                           convolution_descriptor_12,
                                                           output_descriptor_12,
                                                           convolution_algorithm_12,
                                                           &workspace_bytes_12));


        cudaMalloc(&d_workspace_12, workspace_bytes_12);



#endif
        break;
    }
    case CONV2D_13:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_13, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_13, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_13, l.bias_size);

        cudaMemset(d_output_13, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_13,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_13,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_13: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   32x32x64   ----> 32x32x64
        cudnnCreate(&cudnn_13);
        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_13));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_13,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_13));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_13,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_13));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_13,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_13));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_13,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_13,
                                                input_descriptor_13,
                                                kernel_descriptor_13,
                                                convolution_descriptor_13,
                                                output_descriptor_13,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_13));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_13,
                                                           input_descriptor_13,
                                                           kernel_descriptor_13,
                                                           convolution_descriptor_13,
                                                           output_descriptor_13,
                                                           convolution_algorithm_13,
                                                           &workspace_bytes_13));


        cudaMalloc(&d_workspace_13, workspace_bytes_13);



#endif
        break;
    }
    case UPSM2D_2:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_upsm_2, l.output_size);

        cudaMemset(d_output_upsm_2, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD UPSM2D_2: image:%d,%d \n\033[0m",w,h);
        break;
    }

    case CONV2D_14:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_14, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_14, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_14, l.bias_size);

        cudaMemset(d_output_14, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_14,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_14,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_14: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   64x64x64   ----> 64x64x64
        cudnnCreate(&cudnn_14);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_14));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_14,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_14));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_14,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_14));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_14,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_14));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_14,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_14,
                                                input_descriptor_14,
                                                kernel_descriptor_14,
                                                convolution_descriptor_14,
                                                output_descriptor_14,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_14));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_14,
                                                           input_descriptor_14,
                                                           kernel_descriptor_14,
                                                           convolution_descriptor_14,
                                                           output_descriptor_14,
                                                           convolution_algorithm_14,
                                                           &workspace_bytes_14));


        cudaMalloc(&d_workspace_14, workspace_bytes_14);



#endif

        break;
    }
    case CONCAT_2:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_concat_2, l.output_size);

        cudaMemset(d_output_concat_2, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD CONCAT_2: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_15:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_15, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_15, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_15, l.bias_size);

        cudaMemset(d_output_15, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_15,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_15,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_15: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   64x64x128   ----> 64x64x64
        cudnnCreate(&cudnn_15);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_15));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_15,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/128,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_15));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_15,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_15));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_15,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/128,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_15));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_15,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_15,
                                                input_descriptor_15,
                                                kernel_descriptor_15,
                                                convolution_descriptor_15,
                                                output_descriptor_15,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_15));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_15,
                                                           input_descriptor_15,
                                                           kernel_descriptor_15,
                                                           convolution_descriptor_15,
                                                           output_descriptor_15,
                                                           convolution_algorithm_15,
                                                           &workspace_bytes_15));


        cudaMalloc(&d_workspace_15, workspace_bytes_15);

#endif
        break;
    }
    case CONV2D_16:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_16, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_16, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_16, l.bias_size);

        cudaMemset(d_output_16, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_16,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_16,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_16: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   64x64x64   ----> 64x64x64
        cudnnCreate(&cudnn_16);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_16));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_16,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_16));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_16,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_16));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_16,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/64,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_16));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_16,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_16,
                                                input_descriptor_16,
                                                kernel_descriptor_16,
                                                convolution_descriptor_16,
                                                output_descriptor_16,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_16));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_16,
                                                           input_descriptor_16,
                                                           kernel_descriptor_16,
                                                           convolution_descriptor_16,
                                                           output_descriptor_16,
                                                           convolution_algorithm_16,
                                                           &workspace_bytes_16));


        cudaMalloc(&d_workspace_16, workspace_bytes_16);



#endif

        break;
    }
    case UPSM2D_3:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_upsm_3, l.output_size);

        cudaMemset(d_output_upsm_3, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD UPSM2D_3: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_17:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_17, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_17, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_17, l.bias_size);

        cudaMemset(d_output_17, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_17,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_17,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_17: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   128x128x64   ----> 128x128x32
        cudnnCreate(&cudnn_17);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_17));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_17,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_17));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_17,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_17));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_17,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/32,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_17));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_17,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_17,
                                                input_descriptor_17,
                                                kernel_descriptor_17,
                                                convolution_descriptor_17,
                                                output_descriptor_17,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_17));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_17,
                                                           input_descriptor_17,
                                                           kernel_descriptor_17,
                                                           convolution_descriptor_17,
                                                           output_descriptor_17,
                                                           convolution_algorithm_17,
                                                           &workspace_bytes_17));


        cudaMalloc(&d_workspace_17, workspace_bytes_17);



#endif
        break;
    }
    case CONCAT_3:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_concat_3, l.output_size);

        cudaMemset(d_output_concat_3, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD CONCAT_3: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_18:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_18, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_18, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_18, l.bias_size);

        cudaMemset(d_output_18, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_18,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_18,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_18: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   128x128x64   ----> 128x128x32
        cudnnCreate(&cudnn_18);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_18));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_18,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/64,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_18));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_18,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_18));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_18,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/32,
                                              /*in_channels=*/64,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_18));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_18,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_18,
                                                input_descriptor_18,
                                                kernel_descriptor_18,
                                                convolution_descriptor_18,
                                                output_descriptor_18,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_18));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_18,
                                                           input_descriptor_18,
                                                           kernel_descriptor_18,
                                                           convolution_descriptor_18,
                                                           output_descriptor_18,
                                                           convolution_algorithm_18,
                                                           &workspace_bytes_18));


        cudaMalloc(&d_workspace_18, workspace_bytes_18);



#endif
        break;
    }
    case CONV2D_19:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_19, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_19, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_19, l.bias_size);

        cudaMemset(d_output_19, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_19,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_19,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_19: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   128x128x32   ----> 128x128x32
        cudnnCreate(&cudnn_19);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_19));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_19,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_19));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_19,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_19));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_19,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/32,
                                              /*in_channels=*/32,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_19));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_19,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_19,
                                                input_descriptor_19,
                                                kernel_descriptor_19,
                                                convolution_descriptor_19,
                                                output_descriptor_19,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_19));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_19,
                                                           input_descriptor_19,
                                                           kernel_descriptor_19,
                                                           convolution_descriptor_19,
                                                           output_descriptor_19,
                                                           convolution_algorithm_19,
                                                           &workspace_bytes_19));


        cudaMalloc(&d_workspace_19, workspace_bytes_19);



#endif
        break;
    }
    case UPSM2D_4:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_upsm_4, l.output_size);

        cudaMemset(d_output_upsm_4, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD UPSM2D_4: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_20:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_20, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_20, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_20, l.bias_size);

        cudaMemset(d_output_20, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_20,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_20,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_20: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   256x256x32   ----> 256x256x16
        cudnnCreate(&cudnn_20);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_20));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_20,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_20));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_20,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_20));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_20,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/16,
                                              /*in_channels=*/32,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_20));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_20,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_20,
                                                input_descriptor_20,
                                                kernel_descriptor_20,
                                                convolution_descriptor_20,
                                                output_descriptor_20,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_20));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_20,
                                                           input_descriptor_20,
                                                           kernel_descriptor_20,
                                                           convolution_descriptor_20,
                                                           output_descriptor_20,
                                                           convolution_algorithm_20,
                                                           &workspace_bytes_20));


        cudaMalloc(&d_workspace_20, workspace_bytes_20);



#endif
        break;
    }
    case CONCAT_4:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_concat_4, l.output_size);

        cudaMemset(d_output_concat_4, 0, l.output_size);

        l.im_h=h;
        l.im_w=w;

        printf("\033[1;31mLOAD CONCAT_4: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case CONV2D_21:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_21, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_21, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_21, l.bias_size);

        cudaMemset(d_output_21, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_21,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_21,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_20: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   256x256x32   ----> 256x256x16
        cudnnCreate(&cudnn_21);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_21));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_21,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/32,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_21));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_21,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_21));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_21,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/16,
                                              /*in_channels=*/32,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_21));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_21,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_21,
                                                input_descriptor_21,
                                                kernel_descriptor_21,
                                                convolution_descriptor_21,
                                                output_descriptor_21,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_21));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_21,
                                                           input_descriptor_21,
                                                           kernel_descriptor_21,
                                                           convolution_descriptor_21,
                                                           output_descriptor_21,
                                                           convolution_algorithm_21,
                                                           &workspace_bytes_21));


        cudaMalloc(&d_workspace_21, workspace_bytes_21);



#endif
        break;
    }
    case CONV2D_22:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_22, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_22, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_22, l.bias_size);

        cudaMemset(d_output_22, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_22,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_22,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_20: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   256x256x16   ----> 256x256x16
        cudnnCreate(&cudnn_22);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_22));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_22,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_22));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_22,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_22));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_22,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/16,
                                              /*in_channels=*/16,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_22));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_22,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_22,
                                                input_descriptor_22,
                                                kernel_descriptor_22,
                                                convolution_descriptor_22,
                                                output_descriptor_22,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_22));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_22,
                                                           input_descriptor_22,
                                                           kernel_descriptor_22,
                                                           convolution_descriptor_22,
                                                           output_descriptor_22,
                                                           convolution_algorithm_22,
                                                           &workspace_bytes_22));


        cudaMalloc(&d_workspace_22, workspace_bytes_22);



#endif
        break;
    }
    case CONV2D_23:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_23, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_23, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_23, l.bias_size);

        cudaMemset(d_output_23, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_23,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_23,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_20: image:%d,%d \n\033[0m",w,h);
#if using_cudnn==1
        //   256x256x16   ----> 256x256x2
        cudnnCreate(&cudnn_23);


        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_23));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_23,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/16,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_23));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_23,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/2,
                                              /*image_height=*/h,
                                              /*image_width=*/w));


        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_23));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_23,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NCHW,
                                              /*out_channels=*/2,
                                              /*in_channels=*/16,
                                              /*kernel_height=*/3,
                                              /*kernel_width=*/3));


        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_23));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_23,
                                                   /*pad_height=*/1,
                                                   /*pad_width=*/1,
                                                   /*vertical_stride=*/1,
                                                   /*horizontal_stride=*/1,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));


        checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn_23,
                                                input_descriptor_23,
                                                kernel_descriptor_23,
                                                convolution_descriptor_23,
                                                output_descriptor_23,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm_23));


        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_23,
                                                           input_descriptor_23,
                                                           kernel_descriptor_23,
                                                           convolution_descriptor_23,
                                                           output_descriptor_23,
                                                           convolution_algorithm_23,
                                                           &workspace_bytes_23));


        cudaMalloc(&d_workspace_23, workspace_bytes_23);



#endif
        break;
    }
    case CONV2D_24:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMallocHost(&d_output_24, l.output_size);

        l.kernel_size = l.width * l.height * l.nfilters * l.depth * sizeof(float);
        cudaMalloc( (void**)&d_kernel_24, l.kernel_size);

        l.bias_size = l.nfilters * sizeof(float);
        cudaMalloc((void**)&d_bias_24, l.bias_size);

        cudaMemset(d_output_24, 0, l.output_size);
        CHECK(cudaMemcpy(d_kernel_24,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_24,l.bias,l.bias_size,cudaMemcpyHostToDevice));

        cudaMallocHost(&h_output,l.output_size);

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_20: image:%d,%d \n\033[0m",w,h);
        break;
    }
    default:
        break;
    }


}

extern "C" void Remove_NN()
{
    //first layer
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);
    // second layer
    cudaFree(d_output_2);
    cudaFree(d_kernel_2);
    cudaFree(d_bias_2);
    //third layer
    cudaFree(d_output_maxp_1);
    // 4th layer
    cudaFree(d_output_3);
    cudaFree(d_kernel_3);
    cudaFree(d_bias_3);
    // 5th layer
    cudaFree(d_output_4);
    cudaFree(d_kernel_4);
    cudaFree(d_bias_4);
    // 6th layer
    cudaFree(d_output_maxp_2);
    // 7th layer
    cudaFree(d_output_5);
    cudaFree(d_kernel_5);
    cudaFree(d_bias_5);
    // 8th layer
    cudaFree(d_output_6);
    cudaFree(d_kernel_6);
    cudaFree(d_bias_6);
    // 9th layer
    cudaFree(d_output_maxp_3);
    // 10th layer
    cudaFree(d_output_7);
    cudaFree(d_kernel_7);
    cudaFree(d_bias_7);
    // 11th layer
    cudaFree(d_output_8);
    cudaFree(d_kernel_8);
    cudaFree(d_bias_8);
    // 12th layer
    cudaFree(d_output_maxp_4);
    // 13th layer
    cudaFree(d_output_9);
    cudaFree(d_kernel_9);
    cudaFree(d_bias_9);
    // 14th layer
    cudaFree(d_output_10);
    cudaFree(d_kernel_10);
    cudaFree(d_bias_10);
    // 16th layer
    cudaFree(d_output_upsm_1);
    // 17th layer
    cudaFree(d_output_11);
    cudaFree(d_kernel_11);
    cudaFree(d_bias_11);
    // 18th layer
    cudaFree(d_output_concat_1);
    // 20th layer
    cudaFree(d_output_12);
    cudaFree(d_kernel_12);
    cudaFree(d_bias_12);
    // 21th layer
    cudaFree(d_output_13);
    cudaFree(d_kernel_13);
    cudaFree(d_bias_13);
    // 22th layer
    cudaFree(d_output_upsm_2);
    // 23th layer
    cudaFree(d_output_14);
    cudaFree(d_kernel_14);
    cudaFree(d_bias_14);
    // 24th layer
    cudaFree(d_output_concat_2);
    // 25th layer
    cudaFree(d_output_15);
    cudaFree(d_kernel_15);
    cudaFree(d_bias_15);
    // 26th layer
    cudaFree(d_output_16);
    cudaFree(d_kernel_16);
    cudaFree(d_bias_16);
    // 27th layer
    cudaFree(d_output_upsm_3);
    // 28th layer
    cudaFree(d_output_17);
    cudaFree(d_kernel_17);
    cudaFree(d_bias_17);
    // 29th layer
    cudaFree(d_output_concat_3);
    // 30th layer
    cudaFree(d_output_18);
    cudaFree(d_kernel_18);
    cudaFree(d_bias_18);
    // 31th layer
    cudaFree(d_output_19);
    cudaFree(d_kernel_19);
    cudaFree(d_bias_19);
    // 32th layer
    cudaFree(d_output_upsm_4);
    // 33th layer
    cudaFree(d_output_20);
    cudaFree(d_kernel_20);
    cudaFree(d_bias_20);
    // 34th layer
    cudaFree(d_output_concat_4);
    // 35th layer
    cudaFree(d_output_21);
    cudaFree(d_kernel_21);
    cudaFree(d_bias_21);
    // 36th layer
    cudaFree(d_output_22);
    cudaFree(d_kernel_22);
    cudaFree(d_bias_22);
    // 37th layer
    cudaFree(d_output_23);
    cudaFree(d_kernel_23);
    cudaFree(d_bias_23);
    // 38th layer
    cudaFree(d_output_24);
    cudaFree(d_kernel_24);
    cudaFree(d_bias_24);

    cudaFreeHost(h_output);
    printf("\033[1;31mRemove weights from Memory\n\033[0m");
}
