#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "layer.h"
#define DEBUG_MODE 0
#define CPU_DEBUG_MODE 0
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
    CONV2D_10
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
                           int nx, int ny,int layer) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix;
    if (ix < nx && iy < ny)
    {
        float sum=0;
        int which_layer=layer*9;
        for (char r = -1; r < 2; ++r)
        {
            for (char c = -1; c < 2; ++c)
            {
                unsigned int idx = (iy+r)*nx + (ix+c);
                if(!(iy+r==-1 | ix+c==-1 | ix+c==nx | iy+r==ny ))
                {
                    sum+=input_image[idx]*Kernel[3*(r+1)+(c+1)+which_layer];
                    //#if DEBUG_MODE==1
                    //                    printf("row:%d col:%d Pixel:%f Kernel:%f SUM:%f\n",iy+r,
                    //                           ix+c,input_image[idx],Kernel[3*(r+1)+(c+1)+layer*9],sum);
                    //#endif
                }
            }
        }
        output_image[idx_base+nx*ny*layer]=relu(sum+bias[layer]);
    }
}
__global__ void CONV2D_2_GPU1(float *input_image, float *output_image, float *Kernel,float *bias,
                              int nx, int ny,int layer,int depth)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx_base = iy*nx + ix;
    if (ix < nx && iy < ny /*&& ix==0 && iy==0*/)
    {
        float sum=0;
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
#if DEBUG_MODE==1
                        printf("depth:%d row:%d col:%d Pixel:%f Kernel:%f SUM:%f\n",dth,iy+r,
                               ix+c,input_image[idx],Kernel[3*(r+1)+(c+1)+dth*9+layer*9*16],sum);
#endif
                    }
                }
            }
#if DEBUG_MODE==1
            printf("\033[1;31m-------------------------------------------------\033[0m\n");
#endif
        }
        output_image[idx_base+nx*ny*layer]=relu(sum+bias[layer]);
#if DEBUG_MODE==1
        printf("\033[1;32msum:%f\033[0m\n",output_image[idx_base+nx*ny*layer]);
#endif
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
extern "C" void conv2d_1(float* img_ptr,float** output,int w,int h,layer l)
{
    time_sum=0;
    cudaMemcpy(d_input, img_ptr, l.input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias,l.bias,l.bias_size,cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)
    {
        CONV2DGPU1<<<grid,block>>>(d_input,d_output,d_kernel,d_bias,w,h,i);
    }
    printf("time elapsed \033[1;33mconv2d_1:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output, l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;

}

extern "C" void conv2d_2(float** output, int w, int h, layer l)
{
#if CPU_DEBUG_MODE==1
    for (int i = 0; i < 10; ++i)
    {
        printf("line:%d conv2d_2 output[%d]:%f\n",__LINE__,i,img_ptr[i*w*h]);
    }
#endif
    cudaMemset(d_output_2, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_2,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_2,l.bias,l.bias_size,cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        CONV2D_2_GPU1<<<grid,block>>>(d_output,d_output_2,d_kernel_2,d_bias_2,w,h,i,l.depth);
    }
    printf("time elapsed \033[1;33mconv2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_2,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void maxp2d_1(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_maxp_1, 0, l.output_size);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_2,d_output_maxp_1,w,h,i);
    }
    printf("time elapsed \033[1;33mmaxp2d_1:%f msec \n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_maxp_1,l.output_size, cudaMemcpyDeviceToHost);
    *output=h_output;
}
extern "C" void conv2d_3(float** output, int w, int h, layer l)
{
#if CPU_DEBUG_MODE==1
    for (int i = 0; i < 10; ++i)
    {
        printf("line:%d conv2d_2 output[%d]:%f\n",__LINE__,i,img_ptr[i*w*h]);
    }
#endif
    cudaMemset(d_output_3, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_3,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_3,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_3:%f msec \n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_3,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void conv2d_4(float** output, int w, int h, layer l)
{
#if CPU_DEBUG_MODE==1
    for (int i = 0; i < 10; ++i)
    {
        printf("line:%d conv2d_2 output[%d]:%f\n",__LINE__,i,img_ptr[i*w*h]);
    }
#endif
    cudaMemset(d_output_4, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_4,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_4,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_4:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_4,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void maxp2d_2(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_maxp_2, 0, l.output_size);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_4,d_output_maxp_2,w,h,i);
    }
    printf("time elapsed \033[1;33mmaxp2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_maxp_2,l.output_size, cudaMemcpyDeviceToHost);
    *output=h_output;
}
extern "C" void conv2d_5(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_5, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_5,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_5,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_5:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_5,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void conv2d_6(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_6, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_6,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_6,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_6:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_6,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void maxp2d_3(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_maxp_3, 0, l.output_size);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_6,d_output_maxp_3,w,h,i);
    }
    printf("time elapsed \033[1;33mmaxp2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_maxp_3,l.output_size, cudaMemcpyDeviceToHost);
    *output=h_output;
}

extern "C" void conv2d_7(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_7, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_7,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_7,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_7:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_7,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void conv2d_8(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_8, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_8,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_8,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_7:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_8,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void maxp2d_4(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_maxp_4, 0, l.output_size);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    double t1=GetTime();
    for (int i = 0; i < l.nfilters; ++i)   //which volume is selected for conv
    {
        MAXP2D_GPU<<<grid,block>>>(d_output_8,d_output_maxp_4,w,h,i);
    }
    printf("time elapsed \033[1;33mmaxp2d_2:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_maxp_4,l.output_size, cudaMemcpyDeviceToHost);
    *output=h_output;
}

extern "C" void conv2d_9(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_9, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_9,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_9,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_9:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_9,l.output_size, cudaMemcpyDeviceToHost);

    *output=h_output;
}

extern "C" void conv2d_10(float** output, int w, int h, layer l)
{
    cudaMemset(d_output_10, 0, l.output_size);
    CHECK(cudaMemcpy(d_kernel_10,l.weight,l.kernel_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_10,l.bias,l.bias_size,cudaMemcpyHostToDevice));

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
    printf("time elapsed \033[1;33mconv2d_10:%f msec\n\033[0m",1000*(GetTime()-t1));
    float* h_output =(float*)malloc(l.output_size);
    cudaMemcpy(h_output, d_output_10,l.output_size, cudaMemcpyDeviceToHost);

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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_2: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case MAXP2D_1:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_1, l.output_size);

        l.im_h=h;
        l.im_w=w;

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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_3: image:%d,%d \n\033[0m",w,h);
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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_4: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case MAXP2D_2:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_2, l.output_size);

        l.im_h=h;
        l.im_w=w;

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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_5: image:%d,%d \n\033[0m",w,h);
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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_6: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case MAXP2D_3:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_3, l.output_size);

        l.im_h=h;
        l.im_w=w;

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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_7: image:%d,%d \n\033[0m",w,h);
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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_8: image:%d,%d \n\033[0m",w,h);
        break;
    }
    case MAXP2D_4:
    {
        l.output_size = w * h *l.nfilters * sizeof(float);
        cudaMalloc(&d_output_maxp_4, l.output_size);

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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_9: image:%d,%d \n\033[0m",w,h);
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

        l.im_h=h;
        l.im_w=w;
        printf("\033[1;31mLOAD CONV2D_10: image:%d,%d \n\033[0m",w,h);
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
    printf("\033[1;31mRemove weights from Memory\n\033[0m");
}
