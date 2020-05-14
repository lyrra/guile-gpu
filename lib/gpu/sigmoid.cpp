/*
  Usage for?
  hipStreamSynchronize(0);
  hipDeviceSynchronize();
*/
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "hip/hip_runtime.h"

#if !defined(GPU_NUM_CU)
# error "must define GPU_NUM_CU, number of AMD compute-units in your GPU-card"
#endif
#define GPU_NUM_CU_THREADS 256 // number of streaming-processors per SIMD (or total lanes), CU = 4 x SIMD, where each SIMD takes 4x16 lanes

void calc_threads_blocks (int len, int tpb, int *num_blocks, int *threads_per_block)
{
  *threads_per_block = GPU_NUM_CU_THREADS;
  *num_blocks = (len + GPU_NUM_CU_THREADS - 1) / GPU_NUM_CU_THREADS;
}

__global__ void
kernel_f32_sigmoid(float* __restrict__ a, const float* __restrict__ b, size_t lena, size_t lenb)
{
  size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (i < lena) {
    a[i] = 1.0 / (1.0 + __expf(-b[i])); // fast version of expf
  }
}

int f32_sigmoid (float *deviceA, float *deviceB, int lena, int tpb)
{
  int num_blocks;
  int threads_per_block;
  calc_threads_blocks(lena, tpb, &num_blocks, &threads_per_block);
  hipLaunchKernelGGL(kernel_f32_sigmoid,
                     dim3(num_blocks),
                     dim3(threads_per_block),
                     0, 0, deviceA, deviceB, lena, 0);
  return 0;
}

__global__ void
kernel_f32_grad_sigmoid(float* __restrict__ a, const float* __restrict__ b, size_t lena, size_t lenb)
{
  size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (i < lena) {
    float x = 1.0 / (1.0 + __expf(-b[i]));
    a[i] = x * (1.0 - x);
  }
}

int f32_grad_sigmoid (float *deviceA, float *deviceB, int lena, int tpb)
{
  int num_blocks;
  int threads_per_block;
  calc_threads_blocks(lena, tpb, &num_blocks, &threads_per_block);
  hipLaunchKernelGGL(kernel_f32_grad_sigmoid,
                     dim3(num_blocks),
                     dim3(threads_per_block),
                     0, 0, deviceA, deviceB, lena, 0);
  return 0;
}
