
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH  256
#define HEIGHT 32

#define NUM (WIDTH*HEIGHT)

using namespace std;

int f32_sigmoid (float *deviceA, float *deviceB, int width, int height);

int main() {

  float* hostA;
  float* hostB;

  float* deviceA;
  float* deviceB;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

  cout << "hip Device prop succeeded " << endl ;


  int i;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = ((float)i / NUM) * 40 - 20;
  }

  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));

  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));

  f32_sigmoid(deviceA, deviceB, WIDTH, HEIGHT);

  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  for (i = 0; i < NUM; i++) {
    cout << "x: " << i << ", " << hostB[i] << " s: " << hostA[i] << endl;
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));

  free(hostA);
  free(hostB);

  return 0;
}
