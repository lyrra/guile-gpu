/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
//#include <algorithm>
#include <stdlib.h>
#include "hip/hip_runtime.h"

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void
kernel_f32_sigmoid(float* __restrict__ a, const float* __restrict__ b, int width, int height)
{
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  int i = y * width + x;
  if ( i < (width * height)) {
    //a[i] = 1.0 / (1.0 + expf(-b[i]));
    a[i] = 1.0 / (1.0 + __expf(-b[i])); // fast version of expf
  }
}

int f32_sigmoid (float *deviceA, float *deviceB, int width, int height)
{
  hipLaunchKernelGGL(kernel_f32_sigmoid,
                     dim3(width/THREADS_PER_BLOCK_X, height/THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                     0, 0, deviceA, deviceB, width, height);
  return 0;
}

__global__ void
kernel_f32_grad_sigmoid(float* __restrict__ a, const float* __restrict__ b, int width, int height)
{
 
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  int i = y * width + x;
  if ( i < (width * height)) {
    float x = 1.0 / (1.0 + __expf(-b[i]));
    a[i] = x * (1.0 - x);
  }
}

int f32_grad_sigmoid (float *deviceA, float *deviceB, int width, int height)
{

  hipLaunchKernelGGL(kernel_f32_grad_sigmoid,
                     dim3(width/THREADS_PER_BLOCK_X, height/THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                     0, 0, deviceA, deviceB, width, height);
  return 0;
}

