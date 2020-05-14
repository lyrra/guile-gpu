
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

int f32_sigmoid (float *deviceA, float *deviceB, int lena, void *HipStream);

static float sigmoid (float x) {
  return 1.0 / (1.0 + exp(-x));
}

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
  f32_sigmoid(deviceA, deviceB, WIDTH*HEIGHT, NULL);
  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  int fail = 0;
  for (i = 0; i < NUM; i++) {
    float err = sigmoid(hostB[i]) - hostA[i];
    cout << "x: " << i << ", " << hostB[i] << " s: " << hostA[i] << " err: " << err << endl;
    if(abs(err) > 0.001){
      fail = 1;
    }
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));

  free(hostA);
  free(hostB);

  if (fail) {
    cerr << "test-fail" << endl;
    return 1;
  } else {
    return 0;
  }
}
