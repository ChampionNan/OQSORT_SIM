//
//  main.cpp
//  OQSORT
//
//  Created by ChampionNan on 31/8/2022.
//

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <bitset>
#include <random>
#include <chrono>
#include <utility>
#include <fstream>
#include <algorithm>
#include <iomanip>

#define N 671088640//10000000
#define M 33554432 // int type memory restriction
#define NUM_STRUCTURES 10
// #define MEM_IN_ENCLAVE 5
#define DUMMY 0xffffffff
#define NULLCHAR '\0'
#define MY_RAND_MAX 2147483647

#define ALPHA 0.033
#define BETA 0.014
#define P 21

#define BLOCK_DATA_SIZE 4

// OCALL
void ocall_print_string(const char *str);
void OcallReadBlock(int index, int* buffer, size_t blockSize, int structureId);
void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId);
void freeAllocate(int structureIdM, int structureIdF, int size);
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write);
// OQSORT
int myrand();
int Hypergeometric(int NN, int Msize, int n_prime);
void shuffle(int *array, int n);
int SampleTight(int inStructureId, int samplesId, int *trustedM2);
int SampleLoose(int inStructureId, int samplesId, int *trustedM2);
int* quantileCal(int *samples, int start, int end, int p);
int BSFirstGE(int *array, int size, int target);
int BSFirstL(int *array, int size, int target);
std::pair<int, int> MultiLevelPartition(int inStructureId, int *samples, int sampleSize, int p, int outStructureId1);
int ObliviousTightSort(int inStructureId, int inSize, int sampleId, int outStructureId1, int outStructureId2);
std::pair<int, int> ObliviousLooseSort(int inStructureId, int inSize, int sampleId, int outStructureId1, int outStructureId2);

void callSort(int sortId, int structureId, int paddedSize, int *resId, int *resN);
// SUPPORT
void padWithDummy(int structureId, int start, int realNum, int secSize);
int moveDummy(int *a, int size);
void swapRow(int *a, int *b);
bool cmpHelper(int *a, int *b);
int partition(int *arr, int low, int high);
void quickSort(int *arr, int low, int high);
void init(int **arrayAddr, int structurId, int size);
void print(int* array, int size);
void print(int **arrayAddr, int structureId, int size);
void test(int **arrayAddr, int structureId, int size);
void testWithDummy(int **arrayAddr, int structureId, int size);

int *X;
//structureId=3, write back array
int *Y;
int *arrayAddr[NUM_STRUCTURES];
int paddedSize;

int IOcost = 0;
int sampleFlag = 0;
int sampleCost = 0;
int partitionFlag = 0;
int partitionCost = 0;
int finalFlag = 0;
int finalCost = 0;
// TODO: set up structure size
const int structureSize[NUM_STRUCTURES] = {sizeof(int),
  2 * sizeof(int), 2 * sizeof(int),
  sizeof(int), sizeof(int), sizeof(int), sizeof(int)};


/* OCall functions */
void ocall_print_string(const char *str) {
  /* Proxy/Bridge will check the length and null-terminate
   * the input string to prevent buffer overflow.
   */
  printf("%s", str);
  fflush(stdout);
}

void OcallReadBlock(int index, int* buffer, size_t blockSize, int structureId) {
  if (blockSize == 0) {
    // printf("Unknown data size");
    return;
  }
  // memcpy(buffer, arrayAddr[structureId] + index, blockSize * structureSize[structureId]);
  memcpy(buffer, arrayAddr[structureId] + index, blockSize);
  IOcost += 1;
  if (sampleFlag) sampleCost += 1;
  if (partitionFlag) partitionCost += 1;
  if (finalFlag) finalCost += 1;
}

void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId) {
  if (blockSize == 0) {
    // printf("Unknown data size");
    return;
  }
  // memcpy(arrayAddr[structureId] + index, buffer, blockSize * structureSize[structureId]);
  memcpy(arrayAddr[structureId] + index, buffer, blockSize);
  IOcost += 1;
  if (sampleFlag) sampleCost += 1;
  if (partitionFlag) partitionCost += 1;
  if (finalFlag) finalCost += 1;
}


/* main function */
int main(int argc, const char* argv[]) {
  int ret = 1;
  int *resId = (int*)malloc(sizeof(int));
  int *resN = (int*)malloc(sizeof(int));
  // oe_result_t result;
  // oe_enclave_t* enclave = NULL;
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::seconds duration;
  srand((unsigned)time(NULL));
  
  // 0: OSORT-Tight, 1: OSORT-Loose, 2: bucketOSort, 3: bitonicSort
  int sortId = 0;
  int inputId = 0;

  // step1: init test numbers
  inputId = 3;
  X = (int *)malloc(N * sizeof(int));
  arrayAddr[inputId] = X;
  paddedSize = N;
  init(arrayAddr, inputId, paddedSize);

  // step2: Create the enclave
  // print(arrayAddr, inputId, N);
  
  // step3: call sort algorithms
  start = std::chrono::high_resolution_clock::now();
  std::cout << "Test OQSort... " << std::endl;
  callSort(sortId, inputId, paddedSize, resId, resN);
  std::cout << "Result ID: " << *resId << std::endl;
  if (sortId == 0) {
      test(arrayAddr, *resId, paddedSize);
      *resN = N;
  } else {
    // Sample Loose has different test & print
    testWithDummy(arrayAddr, *resId, *resN);
  }
  end = std::chrono::high_resolution_clock::now();
  // step4: std::cout execution time
  duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Finished. Duration Time: " << duration.count() << " seconds" << std::endl;
  std::cout << "Total Cost: " << 1.0 * IOcost * BLOCK_DATA_SIZE / N << ", " << 1.0 * (sampleCost + partitionCost + finalCost) * BLOCK_DATA_SIZE / N <<std::endl;
  std::cout << "Sample Cost: " << 1.0 * sampleCost * BLOCK_DATA_SIZE / N << std::endl;
  std::cout << "Partition Cost: " << 1.0 * partitionCost * BLOCK_DATA_SIZE / N << std::endl;
  std::cout << "Final Cost: " << 1.0 * finalCost * BLOCK_DATA_SIZE / N << std::endl;
  std::cout<<std::fixed<<std::setprecision(2);
  std::cout << 1.0 * IOcost * BLOCK_DATA_SIZE / N << "," << 1.0 * sampleCost * BLOCK_DATA_SIZE / N << "," << 1.0 * partitionCost * BLOCK_DATA_SIZE / N << "," << 1.0 * finalCost * BLOCK_DATA_SIZE / N << std::endl;
  // print(arrayAddr, *resId, *resN);
  // step5: exix part
  exit:
    
    for (int i = 0; i < NUM_STRUCTURES; ++i) {
      if (arrayAddr[i]) {
        free(arrayAddr[i]);
      }
    }
    free(resId);
    free(resN);
    return ret;
}


// TODO: Set this function as OCALL
void freeAllocate(int structureIdM, int structureIdF, int size) {
  // 1. Free arrayAddr[structureId]
  if (arrayAddr[structureIdF]) {
    free(arrayAddr[structureIdF]);
  }
  // 2. malloc new asked size (allocated in outside)
  if (size <= 0) {
    return;
  }
  int *addr = (int*)malloc(size * sizeof(int));
  memset(addr, DUMMY, size * sizeof(int));
  // 3. assign malloc address to arrayAddr
  arrayAddr[structureIdM] = addr;
  return ;
}

// Functions x crossing the enclave boundary, unit: BLOCK_DATA_SIZE
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write) {
  int boundary = (int)((blockSize + BLOCK_DATA_SIZE - 1 )/ BLOCK_DATA_SIZE);
  int Msize;
  int multi = structureSize[structureId] / sizeof(int);
  if (!write) {
    // OcallReadBlock(index, block, blockSize * structureSize[structureId], structureId);
    for (int i = 0; i < boundary; ++i) {
      Msize = std::min(BLOCK_DATA_SIZE, (int)blockSize - i * BLOCK_DATA_SIZE);
      OcallReadBlock(index + multi * i * BLOCK_DATA_SIZE, &block[i * BLOCK_DATA_SIZE * multi], Msize * structureSize[structureId], structureId);
    }
  } else {
    // OcallWriteBlock(index, block, blockSize * structureSize[structureId], structureId);
    for (int i = 0; i < boundary; ++i) {
      Msize = std::min(BLOCK_DATA_SIZE, (int)blockSize - i * BLOCK_DATA_SIZE);
      OcallWriteBlock(index + multi * i * BLOCK_DATA_SIZE, &block[i * BLOCK_DATA_SIZE * multi], Msize * structureSize[structureId], structureId);
    }
  }
  return;
}

// TODO: calculate Hypergeometric Distribution
/*
int Hypergeometric(int NN, int Msize, int n_prime) {
  int m = 0;
  std::random_device rd;
  std::mt19937_64 generator(rd());
  double rate = double(n_prime) / NN;
  std::bernoulli_distribution b(rate);
  for (int j = 0; j < Msize; ++j) {
    if (b(generator)) {
      m += 1;
      n_prime -= 1;
    }
    NN -= 1;
    rate = double(n_prime) / double(NN);
    std::bernoulli_distribution b(rate);
  }
  return m;
}*/
// generate random numbers using uniform distribution
int myrand() {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(0, INT_MAX);
  return dist(mt);
}

int Hypergeometric(int NN, int Msize, int n_prime) {
  int m = 0;
  srand((unsigned)time(0));
  double rate = double(n_prime) / NN;
  for (int j = 0; j < Msize; ++j) {
    if (rand() / double(INT_MAX) < rate) {
      m += 1;
      n_prime -= 1;
    }
    NN -= 1;
    rate = double(n_prime) / double(NN);
  }
  return m;
}

void shuffle(int *array, int n) {
  if (n > 1) {
    for (int i = 0; i < n - 1; ++i) {
      int j = i + rand() / (INT_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

// Return sorted samples
int SampleTight(int inStructureId, int samplesId, int *trustedM2) {
  sampleFlag = 1;
  int N_prime = N;
  int n_prime = ceil(1.0 * ALPHA * N);
  int M2 = BLOCK_DATA_SIZE;
  int boundary = (int)ceil(1.0 * N / M2);
  int Msize, alphaM22;
  int m; // use for hypergeometric distribution
  int realNum = 0; // #pivots
  int readStart = 0;
  int *trustedM1 = (int*)malloc(M2 * sizeof(int));
  
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(M2, N - i * M2);
    alphaM22 = (int)ceil(2.0 * ALPHA * Msize);
    opOneLinearScanBlock(readStart, trustedM1, Msize, inStructureId, 0);
    // print(trustedMemory, Msize);
    readStart += Msize;
    // step1. sample with hypergeometric distribution
    m = Hypergeometric(N_prime, Msize, n_prime);
    /*if (m > alphaM22 && (i != boundary - 1)) {
      return -1;
    }*/
    // step2. shuffle M
    shuffle(trustedM1, Msize);
    // step3. set dummy (REMOVE)
    // step4. write sample back to memory
    memcpy(&trustedM2[realNum], trustedM1, m * sizeof(int));
    realNum += m;
    N_prime -= Msize;
    n_prime -= m;
    if (n_prime <= 0) {
      break;
    }
  }
  
  quickSort(trustedM2, 0, realNum - 1);
  double nonDummyNum = ALPHA * N;
  printf("%d, %f\n", realNum, nonDummyNum);
  free(trustedM1);
  sampleFlag = 0;
  return realNum;
}


int SampleLoose(int inStructureId, int samplesId, int *trustedM2) {
  sampleFlag = 1;
  int N_prime = N;
  int n_prime = ceil(1.0 * ALPHA * N);
  int boundary = (int)ceil(1.0 * N/BLOCK_DATA_SIZE);
  int Msize;
  int m; // use for hypergeometric distribution
  int k = 0;
  int realNum = 0; // #pivots
  int readStart = 0;
  int *trustedMemory = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  
  freeAllocate(samplesId, samplesId, n_prime);
  
  for (int i = 0; i < boundary; i++) {
    // step1. sample with hypergeometric distribution
    Msize = std::min(BLOCK_DATA_SIZE, N - i * BLOCK_DATA_SIZE);
    m = Hypergeometric(N_prime, Msize, n_prime);
    if (m > 0) {
      realNum += m;
      opOneLinearScanBlock(readStart, trustedMemory, Msize, inStructureId, 0);
      readStart += Msize;
      // step2. shuffle M
      shuffle(trustedMemory, Msize);
      // step4. write sample back to external memory
      memcpy(&trustedM2[k], trustedMemory, m * sizeof(int));
      k += m;
      // TODO: n_prime should be placed in if (m > 0)
      n_prime -= m;
    }
    N_prime -= Msize;
    if (n_prime <= 0) {
      break;
    }
    // TODO: ? what's m value
  }
  quickSort(trustedM2, 0, realNum - 1);
  double nonDummyNum = ALPHA * N;
  printf("%d, %f\n", realNum, nonDummyNum);
  free(trustedMemory);
  sampleFlag = 0;
  return realNum;
}

// TODO: here sample size = 2 * n ? [start, end)
int* quantileCal(int *samples, int start, int end, int p) {
  int sampleSize = end - start;
  // use samples[0, p] to store pivots
  for (int i = 1; i < p; i++) {
    // std::cout << "Sample index: " << i * sampleSize / p << std::endl;
    samples[i] = samples[i * sampleSize / p];
  }
  samples[0] = INT_MIN;
  samples[p] = INT_MAX;
  // TODO:
  int *trustedM1 = (int*)realloc(samples, sizeof(int) * (p + 1));
  return trustedM1;
}

int BSFirstGE(int *array, int size, int target) {
  int l = 0, r = size - 1;
  while (l <= r) {
    int mid = l + ((r - l) >> 1);
    if (array[mid] < target) {
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  if (l >= size) {
    return -1;
  }
  return l;
}

int BSFirstL(int *array, int size, int target) {
  int l = 0, r = size - 1;
  while (l <= r) {
    int mid = l + ((r - l) >> 1);
    if (array[mid] >= target) {
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  if (r < 0) {
    return -1;
  }
  return r;
}


std::pair<int, int> MultiLevelPartition(int inStructureId, int *samples, int sampleSize, int p, int outStructureId1) {
  partitionFlag = 1;
  if (N <= M) {
    return std::make_pair(N, 1);
  }
  int hatN = (int)ceil(1.0 * (1 + 2 * BETA) * N);
  int M_prime = (int)ceil(1.0 * M / (1 + 2 * BETA));
  
  // 2. set up block index array L & shuffle (REMOVE)
  int r = (int)ceil(1.0 * log(hatN / M) / log(p));
  int p0 = (int)ceil(1.0 * hatN / (M * pow(p, r - 1)));
  // 3. calculate p0-quantile about sample
  // int *trustedM1 = (int*)malloc(sizeof(int) * (p0 + 1));
  int *trustedM1 = nullptr;
  trustedM1 = quantileCal(samples, 0, sampleSize, p0);
  while (!trustedM1) {
    std::cout << "Quantile calculate error!\n";
    trustedM1 = quantileCal(samples, 0, sampleSize, p0);
  }
  // 4. allocate trusted memory
  int boundary1 = (int)ceil(1.0 * N / M_prime);
  int boundary2 = (int)ceil(1.0 * M_prime / BLOCK_DATA_SIZE);
  int dataBoundary = boundary2 * BLOCK_DATA_SIZE;
  int smallSectionSize = M / p0;
  int bucketSize0 = boundary1 * smallSectionSize;
  // input data
  int *trustedM3 = (int*)malloc(sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  int **bucketNum = (int**)malloc(sizeof(int*) * p0);
  for (int i = 0; i < p0; ++i) {
    bucketNum[i] = (int*)malloc(sizeof(int) * boundary1);
    memset(bucketNum[i], 0, sizeof(int) * boundary1);
  }
  freeAllocate(outStructureId1, outStructureId1, boundary1 * smallSectionSize * p0);
  
  // int Rsize = 0;  // all input number, using for check
  int k, Msize1, Msize2;
  int blocks_done = 0;
  int total_blocks = (int)ceil(1.0 * N / BLOCK_DATA_SIZE);
  int *shuffleB = (int*)malloc(sizeof(int) * BLOCK_DATA_SIZE);
  // 6. level0 partition
  for (int i = 0; i < boundary1; ++i) {
    for (int j = 0; j < boundary2; ++j) {
      if (total_blocks - 1 - blocks_done == 0) {
        k = 0;
      } else {
        k = rand() % (total_blocks - 1 - blocks_done);
      }
      Msize1 = std::min(BLOCK_DATA_SIZE, N - k * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, &trustedM3[j * BLOCK_DATA_SIZE], Msize1, inStructureId, 0);
      memset(shuffleB, DUMMY, sizeof(int) * BLOCK_DATA_SIZE);
      Msize2 = std::min(BLOCK_DATA_SIZE, N - (total_blocks-1-blocks_done) * BLOCK_DATA_SIZE);
      opOneLinearScanBlock((total_blocks-1-blocks_done) * BLOCK_DATA_SIZE, shuffleB, Msize2, inStructureId, 0);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, shuffleB, BLOCK_DATA_SIZE, inStructureId, 1);
      blocks_done += 1;
      if (blocks_done == total_blocks) {
        break;
      }
    }
    int blockNum = moveDummy(trustedM3, dataBoundary);
    quickSort(trustedM3, 0, blockNum - 1);
    for (int j = 0; j < p0; ++j) {
      int pivot1 = trustedM1[j], pivot2 = trustedM1[j + 1];
      int index1 = BSFirstGE(trustedM3, blockNum, pivot1);
      int index2 = BSFirstL(trustedM3, blockNum, pivot2);
      int wNum = index2 - index1 + 1;
      opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, &trustedM3[index1], wNum, outStructureId1, 1);
      bucketNum[j][i] += wNum;
      if (bucketNum[j][i] > smallSectionSize) {
        std::cout << "Overflow in small section M/2p0: " << bucketNum[j][i] << std::endl;
      }
    }
    memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  }
  
  // 7. pad dummy
  for (int j = 0; j < p0; ++j) {
    for (int i = 0; i < boundary1; ++i) {
      padWithDummy(outStructureId1, j * bucketSize0 + i * smallSectionSize, bucketNum[j][i], smallSectionSize);
    }
  }
  free(trustedM1);
  free(trustedM3);
  for (int i = 0; i < p0; ++i) {
    free(bucketNum[i]);
  }
  free(bucketNum);
  if (bucketSize0 > M) {
    std::cout << "Each section size is greater than M, adjst parameters: " << bucketSize0 << ", " << M << std::endl;
  }
  // freeAllocate(LIdOut, LIdOut, 0);
  partitionFlag = 0;
  return std::make_pair(bucketSize0, p0);
}


int ObliviousTightSort(int inStructureId, int inSize, int sampleId, int outStructureId1, int outStructureId2) {
  int *trustedM;
  if (inSize <= M) {
    trustedM = (int*)malloc(sizeof(int) * M);
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0);
    quickSort(trustedM, 0, inSize - 1);
    freeAllocate(outStructureId1, outStructureId1, inSize);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1);
    free(trustedM);
    return outStructureId1;
  }
  int M2 = M - BLOCK_DATA_SIZE;
  int *trustedM2 = (int*)malloc(M2 * sizeof(int));
  int realNum = SampleTight(inStructureId, sampleId, trustedM2);
  int n = (int)ceil(1.0 * ALPHA * N);
  while (realNum < 0) {
    std::cout << "Samples number error!\n";
    realNum = SampleTight(inStructureId, sampleId, trustedM2);
  }
  
  std::pair<int, int> section = MultiLevelPartition(inStructureId, trustedM2, std::min(realNum, n), P, outStructureId1);
  int sectionSize = section.first;
  int sectionNum = section.second;
  // int totalLevelSize = sectionNum * sectionSize;
  int j = 0;
  int k;
  
  freeAllocate(outStructureId2, outStructureId2, inSize);
  trustedM = (int*)malloc(sizeof(int) * M);
  finalFlag = 1;
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0);
    // TODO: optimize to utilize bucketNum[j][i]
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k - 1);
    // print(trustedM, k);
    opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1);
    j += k;
    if (j > inSize) {
      std::cout << "Overflow" << std::endl;
    }
  }
  free(trustedM);
  // print(arrayAddr, outStructureId2, inSize);
  finalFlag = 0;
  return outStructureId2;
}

std::pair<int, int> ObliviousLooseSort(int inStructureId, int inSize, int sampleId, int outStructureId1, int outStructureId2) {
  int *trustedM;
  if (inSize <= M) {
    trustedM = (int*)malloc(sizeof(int) * M);
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0);
    quickSort(trustedM, 0, inSize - 1);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1);
    free(trustedM);
    return {outStructureId1, inSize};
  }
  int M2 = M - BLOCK_DATA_SIZE;
  int *trustedM2 = (int*)malloc(M2 * sizeof(int));
  int realNum = SampleLoose(inStructureId, sampleId, trustedM2);
  int n = (int)ceil(1.0 * ALPHA * N);
  while (realNum < 0) {
    std::cout << "Samples number error!\n";
    realNum = SampleLoose(inStructureId, sampleId, trustedM2);
  }

  std::pair<int, int> section = MultiLevelPartition(inStructureId, trustedM2, std::min(realNum, n), P, outStructureId1);
  int sectionSize = section.first;
  int sectionNum = section.second;
  int totalLevelSize = sectionNum * sectionSize;
  int k;
  
  freeAllocate(outStructureId2, outStructureId2, totalLevelSize);
  trustedM = (int*)malloc(sizeof(int) * M);
  finalFlag = 1;
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1);
  }
  finalFlag = 0;
  return {outStructureId2, totalLevelSize};
}


// trusted function
void callSort(int sortId, int structureId, int paddedSize, int *resId, int *resN) {
  // TODO: Change trans-Id
  if (sortId == 0) {
    *resId = ObliviousTightSort(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 3);
  } else if (sortId == 1) {
    std::pair<int, int> ans = ObliviousLooseSort(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 3);
    *resId = ans.first;
    *resN = ans.second;
  }
}


void padWithDummy(int structureId, int start, int realNum, int secSize) {
  int len = secSize - realNum;
  if (len <= 0) {
    return ;
  }
  
  if (structureSize[structureId] == 4) {
    int *junk = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; ++i) {
      junk[i] = DUMMY;
    }
    opOneLinearScanBlock(start + realNum, (int*)junk, len, structureId, 1);
    free(junk);
  
  }
}

int moveDummy(int *a, int size) {
  // k: #elem != DUMMY
  int k = 0;
  for (int i = 0; i < size; ++i) {
    if (a[i] != DUMMY) {
      if (i != k) {
        swapRow(&a[i], &a[k++]);
      } else {
        k++;
      }
    }
  }
  return k;
}


void swapRow(int *a, int *b) {
  int *temp = (int*)malloc(sizeof(int));
  memmove(temp, a, sizeof(int));
  memmove(a, b, sizeof(int));
  memmove(b, temp, sizeof(int));
  free(temp);
}

bool cmpHelper(int *a, int *b) {
  return (*a > *b) ? true : false;
}

int partition(int *arr, int low, int high) {
  // TODO: random version
  // srand(unsigned(time(NULL)));
  int randNum = rand() % (high - low + 1) + low;
  swapRow(arr + high, arr + randNum);
  int *pivot = arr + high;
  int i = low - 1;
  for (int j = low; j <= high - 1; ++j) {
    if (cmpHelper(pivot, arr + j)) {
      i++;
      if (i != j) {
        swapRow(arr + i, arr + j);
      }
    }
  }
  if (i + 1 != high) {
    swapRow(arr + i + 1, arr + high);
  }
  return (i + 1);
}


void quickSort(int *arr, int low, int high) {
  if (high > low) {
    int mid = partition(arr, low, high);
    quickSort(arr, low, mid - 1);
    quickSort(arr, mid + 1, high);
  }
}


/** -------------- SUB-PROCEDURES  ----------------- **/

/** procedure test() : verify sort results **/
void init(int **arrayAddr, int structurId, int size) {
  int i;
  int *addr = (int*)arrayAddr[structurId];
  for (i = 0; i < size; i++) {
    addr[i] = (size - i);
  }
  for(i = size - 1; i >= 1; --i) {
    swapRow(addr + i, addr + (rand() % i));
  }
}

void print(int* array, int size) {
  int i;
  for (i = 0; i < size; i++) {
    printf("%d ", array[i]);
    if ((i != 0) && (i % 5 == 0)) {
      printf("\n");
    }
  }
  printf("\n");
}

void print(int **arrayAddr, int structureId, int size) {
  int i;
  std::ofstream fout("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/output.txt");
  if(structureSize[structureId] == 4) {
    int *addr = (int*)arrayAddr[structureId];
    for (i = 0; i < size; i++) {
      // printf("%d ", addr[i]);
      fout << addr[i] << " ";
      if ((i != 0) && (i % 8 == 0)) {
        // printf("\n");
        fout << std::endl;
      }
    }
  }
  // printf("\n");
  fout << std::endl;
  fout.close();
}

// TODO: change nt types
void test(int **arrayAddr, int structureId, int size) {
  int pass = 1;
  int i;
  // print(structureId);
  if(structureSize[structureId] == 4) {
    for (i = 1; i < size; i++) {
      pass &= ((arrayAddr[structureId])[i-1] < (arrayAddr[structureId])[i]);
      if ((arrayAddr[structureId])[i] == 0) {
        pass = 0;
        break;
      }
    }
  }
  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
}

void testWithDummy(int **arrayAddr, int structureId, int size) {
  int i = 0;
  int j = 0;
  // print(structureId);
  if(structureSize[structureId] == 4) {
    for (i = 0; i < size; ++i) {
      if ((arrayAddr[structureId])[i] != DUMMY) {
        break;
      }
    }
    if (i == size - 1) {
      printf(" TEST PASSed\n");
      return;
    }
    while (i < size && j < size) {
      for (j = i + 1; j < size; ++j) {
        if ((arrayAddr[structureId])[j] != DUMMY) {
          break;
        }
      }
      if (j == size) { // Only 1 element not dummy
        printf(" TEST PASSed\n");
        return;
      }
      if ((arrayAddr[structureId])[i] < (arrayAddr[structureId])[j]) {
        i = j;
      } else {
        printf(" TEST FAILed\n");
        return;
      }
    }
    printf(" TEST PASSed\n");
    return;
  }
}



