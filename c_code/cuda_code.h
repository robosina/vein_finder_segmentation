#ifndef CUDA_CODE_H
#define CUDA_CODE_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
inline void gpu_error_checker(cudaError_t error,const char*file, int line);
#endif // CUDA_CODE_H
