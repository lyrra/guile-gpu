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

void calc_threads_blocks (int len, int tpb, int *num_blocks, int *threads_per_block)
{
  *num_blocks = 1;
  *threads_per_block = 16;
  if (tpb) {
    int new_num_blocks = len / tpb;
    if (new_num_blocks) {
      *num_blocks = new_num_blocks;
      *threads_per_block = tpb;
    }
  }
}

__global__ void
kernel_f32_sigmoid(float* __restrict__ a, const float* __restrict__ b, size_t lena, size_t lenb)
{
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  for(size_t i = offset; i < lena; i++) {
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
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  for(size_t i = offset; i < lena; i++) {
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
