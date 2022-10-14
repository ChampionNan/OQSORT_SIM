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
// TODO: new
#include <unordered_set>
#include <unordered_map>
#include <bitset>
// #include "/usr/local/Cellar/mbedtls/3.1.0/include/mbedtls/aes.h"
// #include "boost/math/distributions/hypergeometric.hpp"


#define N 335544320//10000000
#define M 16777216 // int type memory restriction
#define BLOCK_DATA_SIZE 4
#define NUM_STRUCTURES 10
// #define MEM_IN_ENCLAVE 5
#define DUMMY 0xffffffff
#define NULLCHAR '\0'
#define MY_RAND_MAX 2147483647

#define _ALPHA -1
#define _BETA -1
#define _P -1
#define ALPHA 0.0318303755519756
#define BETA 0.0196472251883635
#define P 21

// OCALL
void ocall_print_string(const char *str);
void OcallReadBlock(int index, int* buffer, size_t blockSize, int structureId);
void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId);
void freeAllocate(int structureIdM, int structureIdF, int size);
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write, int dummyNum);
// OQSORT


void floydSampler(int n, int k, std::vector<int> &x);
int Sample(int inStructureId, int sampleSize, std::vector<int> &trustedM2, int is_tight, int is_rec);
void SampleRec(int inStructureId, int sampleId, int sortedSampleId, int is_tight, std::vector<std::vector<int>>& pivots);
void quantileCal(std::vector<int> &samples, int start, int end, int p);
int partitionMulti(int *arr, int low, int high, int pivot);
void quickSortMulti(int *arr, int low, int high, std::vector<int> pivots, int left, int right, std::vector<int> &partitionIdx);
std::pair<int, int> OneLevelPartition(int inStructureId, int inSize, std::vector<int> &samples, int sampleSize, int p, int outStructureId1, int is_rec);
std::pair<int, int> TwoLevelPartition(int inStructureId, std::vector<std::vector<int>>& pivots, int p, int outStructureId1, int outStructureId2);
int ObliviousTightSort(int inStructureId, int inSize, int outStructureId1, int outStructureId2);
int ObliviousTightSort2(int inStructureId, int inSize, int sampleId, int sortedSampleId, int outStructureId, int outStructureId2);
std::pair<int, int> ObliviouLooseSort(int inStructureId, int inSize, int outStructureId1, int outStructureId2);
std::pair<int, int> ObliviousLooseSort2(int inStructureId, int inSize, int sampleId, int sortedSampleId, int outStructureId1, int outStructureId2);
void ObliviousLooseSortRec(int sampleId, int sampleSize, int sortedSampleId, std::vector<std::vector<int>>& pivots);
// SUPPORT
void prf(unsigned char *right, char key, int tweak, unsigned char *ret);
void round(unsigned char* data, char key, int tweak, unsigned char* newData);
int encrypt(int index, char key[8], int rounds);

void callSort(int sortId, int structureId, int paddedSize, int *resId, int *resN);
int myrand();
int Hypergeometric(int NN, int Msize, int n_prime);
void shuffle(int *array, int n);
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
int is_tight = 1;

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
}

void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId) {
  if (blockSize == 0) {
    // printf("Unknown data size");
    return;
  }
  // memcpy(arrayAddr[structureId] + index, buffer, blockSize * structureSize[structureId]);
  memcpy(arrayAddr[structureId] + index, buffer, blockSize);
  IOcost += 1;
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
  // char key[8] = "=-÷×&";
  // int res = encrypt(3, key, 3);
  // std::cout << res << std::endl;

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
  std::cout << "IOcost: " << 1.0 * IOcost / N * BLOCK_DATA_SIZE << std::endl;
  print(arrayAddr, *resId, *resN);
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
// TODO: FIX This function ? need checking
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write, int dummyNum) {
  if (blockSize + dummyNum == 0) {
    return ;
  }
  int boundary = (int)((blockSize + BLOCK_DATA_SIZE - 1 )/ BLOCK_DATA_SIZE);
  int Msize, i;
  int multi = structureSize[structureId] / sizeof(int);
  if (!write) {
    // OcallReadBlock(index, block, blockSize * structureSize[structureId], structureId);
    for (i = 0; i < boundary; ++i) {
      Msize = std::min(BLOCK_DATA_SIZE, (int)blockSize - i * BLOCK_DATA_SIZE);
      OcallReadBlock(index + multi * i * BLOCK_DATA_SIZE, &block[i * BLOCK_DATA_SIZE * multi], Msize * structureSize[structureId], structureId);
    }
  } else {
    // OcallWriteBlock(index, block, blockSize * structureSize[structureId], structureId);
    for (i = 0; i < boundary; ++i) {
      Msize = std::min(BLOCK_DATA_SIZE, (int)blockSize - i * BLOCK_DATA_SIZE);
      OcallWriteBlock(index + multi * i * BLOCK_DATA_SIZE, &block[i * BLOCK_DATA_SIZE * multi], Msize * structureSize[structureId], structureId);
    }
    if (dummyNum) {
      int *junk = (int*)malloc(dummyNum * multi * sizeof(int));
      for (int j = 0; j < dummyNum * multi; ++j) {
        junk[j] = DUMMY;
      }
      int startIdx = index + multi * blockSize;
      boundary = ceil(1.0 * dummyNum / BLOCK_DATA_SIZE);
      for (int j = 0; j < boundary; ++j) {
        Msize = std::min(BLOCK_DATA_SIZE, dummyNum - j * BLOCK_DATA_SIZE);
        OcallWriteBlock(startIdx + multi * j * BLOCK_DATA_SIZE, &junk[j * BLOCK_DATA_SIZE * multi], Msize * structureSize[structureId], structureId);
      }
    }
  }
  return;
}


void floydSampler(int n, int k, std::vector<int> &x) {
  std::unordered_set<int> H;
  for (int i = n - k; i < n; ++i) {
    x.push_back(i);
  }
  unsigned int seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine e(seed1);
  int r, j, temp;
  for (int i = 0; i < k; ++i) {
    std::uniform_int_distribution<int> dist{0, n-k+1+i};
    r = dist(e); // get random numbers with PRNG
    if (H.count(r)) {
      std::uniform_int_distribution<int> dist2{0, i};
      j = dist2(e);
      temp = x[i];
      x[i] = x[j];
      x[j] = temp;
      H.insert(n-k+i);
    } else {
      x[i] = r;
      H.insert(r);
    }
  }
  sort(x.begin(), x.end());
}

// TODO: Unity Different Sample
int Sample(int inStructureId, int sampleSize, std::vector<int> &trustedM2, int is_tight, int is_rec=0) {
  int N_prime = sampleSize;
  double alpha = (!is_rec) ? ALPHA : _ALPHA;
  int n_prime = ceil(1.0 * alpha * N_prime);
  int boundary = ceil(1.0 * N_prime / BLOCK_DATA_SIZE);
  int j = 0, Msize;
  int *trustedM1 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  std::vector<int> sampleIdx;
  floydSampler(N_prime, n_prime, sampleIdx);
  for (int i = 0; i < boundary; ++i) {
    if (is_tight) {
      Msize = std::min(BLOCK_DATA_SIZE, N_prime - i * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(i * BLOCK_DATA_SIZE, trustedM1, Msize, inStructureId, 0, 0);
      while ((j < n_prime) && (sampleIdx[j] >= i * BLOCK_DATA_SIZE) && (sampleIdx[j] < (i+1) * BLOCK_DATA_SIZE)) {
        trustedM2.push_back(trustedM1[sampleIdx[j] % BLOCK_DATA_SIZE]);
        j += 1;
      }
    } else if ((!is_tight) && (sampleIdx[j] >= i * BLOCK_DATA_SIZE) && (sampleIdx[j] < (i+1) * BLOCK_DATA_SIZE)) {
      Msize = std::min(BLOCK_DATA_SIZE, N_prime - i * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(i * BLOCK_DATA_SIZE, trustedM1, Msize, inStructureId, 0, 0);
      while ((sampleIdx[j] >= i * BLOCK_DATA_SIZE) && (sampleIdx[j] < (i+1) * BLOCK_DATA_SIZE)) {
        trustedM2.push_back(trustedM1[sampleIdx[j] % BLOCK_DATA_SIZE]);
        j += 1;
        if (j >= n_prime) break;
      }
      if (j >= n_prime) break;
    }
  }
  sort(trustedM2.begin(), trustedM2.end());
  return n_prime;
}

// TODO: What's return value?
void SampleRec(int inStructureId, int sampleId, int sortedSampleId, int is_tight, std::vector<std::vector<int>>& pivots) {
  int N_prime = N;
  int n_prime = ceil(1.0 * ALPHA * N_prime);
  int boundary = ceil(1.0 * N / M);
  int realNum = 0;
  int readStart = 0;
  int *trustedM1 = (int*)malloc(M * sizeof(int));
  int m = 0, Msize;
  freeAllocate(sampleId, sampleId, n_prime);
  for (int i = 0; i < boundary; ++i) {
    Msize = std::min(M, N - i * M);
    // TODO: USing boost library
    m = Hypergeometric(N_prime, Msize, n_prime);
    if (is_tight || (!is_tight && m > 0)) {
      opOneLinearScanBlock(readStart, trustedM1, Msize, inStructureId, 0, 0);
      readStart += Msize;
      shuffle(trustedM1, Msize);
      opOneLinearScanBlock(realNum, trustedM1, m, sampleId, 1, 0);
      realNum += m;
      n_prime -= m;
    }
    N_prime -= Msize;
  }
  std::cout << "Till Sample IOcost: " << 1.0*IOcost/N*BLOCK_DATA_SIZE << std::endl;
  if (realNum > M) {
    ObliviousLooseSortRec(sampleId, realNum, sortedSampleId, pivots);
  }
  return ;
}

// TODO: vector quantile
void quantileCal(std::vector<int> &samples, int start, int end, int p) {
  int sampleSize = end - start;
  for (int i = 1; i < p; ++i) {
    samples[i] = samples[i * sampleSize / p];
  }
  samples[0] = INT_MIN;
  samples[p] = INT_MAX;
  samples.resize(p+1);
  samples.shrink_to_fit();
  return ;
}

// TODO: Add new multi-pivots quicksort
int partitionMulti(int *arr, int low, int high, int pivot) {
  int i = low - 1;
  for (int j = low; j < high + 1; ++j) {
    if (pivot > arr[j]) {
      i += 1;
      swapRow(arr + i, arr + j);
    }
  }
  return i;
}

void quickSortMulti(int *arr, int low, int high, std::vector<int> pivots, int left, int right, std::vector<int> &partitionIdx) {
  int pivotIdx, pivot, mid;
  if (right >= left) {
    pivotIdx = (left + right) >> 1;
    pivot = pivots[pivotIdx];
    mid = partitionMulti(arr, low, high, pivot);
    partitionIdx.push_back(mid);
    quickSortMulti(arr, low, mid, pivots, left, pivotIdx-1, partitionIdx);
    quickSortMulti(arr, mid+1, high, pivots, pivotIdx+1, right, partitionIdx);
  }
}

int BSFirstGreaterEqual(std::vector<int> &nums,int target)
{
    int left = 0, right = nums.size() - 1, res = right;
    while(left <= right)
    {
        int mid = left + (right - left) / 2 ;
        if(nums[mid] >= target)
        {
            res = mid; right = mid - 1;
        }else
        {
            left = mid + 1;
        }
 
    }
    if(target<=nums[res]) return res;
    return -1;
}

// Partition num to =target & <target
// Return the first element index greater than target
int partitionEqual(int *num, int size, int target) {
  int i = -1;
  for (int j = 0; j < size; ++j) {
    if (num[j] == target) {
      i++;
      if (i != j) {
        swapRow(num+i, num+j);
      }
    }
  }
  return i;
}

/*
std::pair<int, int> OneLevelPartition(int inStructureId, int inSize, std::vector<int> &samples, int sampleSize, int p, int outStructureId1, int is_rec=0, int is_duplicate=0) {
  if (inSize <= M) {
    return {inSize, 1};
  }
  double beta = (!is_rec) ? BETA : _BETA;
  int hatN = ceil(1.0 * (1 + 2 * beta) * inSize);
  int M_prime = ceil(1.0 * M / (1 + 2 * beta));
  int r = ceil(1.0 * log(hatN / M) / log(p));
  int p0 = ceil(1.0 * hatN / (M * pow(p, r - 1)));
  quantileCal(samples, 0, sampleSize, p0);
  for (int i = 0; i < samples.size(); ++i) {
    std::cout << samples[i] << ' ';
  }
  std::cout << std::endl;
  int boundary1 = ceil(1.0 * inSize / M_prime);
  int boundary2 = ceil(1.0 * M_prime / BLOCK_DATA_SIZE);
  int dataBoundary = boundary2 * BLOCK_DATA_SIZE;
  int smallSectionSize = M / p0;
  int bucketSize0 = boundary1 * smallSectionSize;
  // Memory each section real data num for later duplicate numbers
  freeAllocate(outStructureId1, outStructureId1, boundary1 * smallSectionSize * p0);
  int k, Msize1, Msize2, index1, index2, index3, writeBackNum, equalNum, eachNum;
  int blocks_done = 0;
  int total_blocks = ceil(1.0 * inSize / BLOCK_DATA_SIZE);
  int *trustedM3 = (int*)malloc(sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  int *shuffleB = (int*)malloc(sizeof(int) * BLOCK_DATA_SIZE);
  std::vector<int> partitionIdx;
  std::unordered_map<int, int> dupPivots;
  // Add pivots map: record #pivots & corresponding elements
  for (int a = 1; a < p0; ++a) {
    if (dupPivots.count(samples[a]) > 0) {
      dupPivots[samples[a]] += 1;
    } else {
      dupPivots.insert({samples[a], 1});
    }
  }
  // TODO: count each memory block # duplicate keys, find out why misssing some pivot numbers
  // TODO: Find FFSEM implementation in c++
  for (int i = 0; i < boundary1; ++i) {
    for (int j = 0; j < boundary2; ++j) {
      if (total_blocks - 1 - blocks_done == 0) {
        k = 0;
      } else {
        k = rand() % (total_blocks - blocks_done);
      }
      Msize1 = std::min(BLOCK_DATA_SIZE, inSize - k * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, &trustedM3[j*BLOCK_DATA_SIZE], Msize1, inStructureId, 0, 0);
      memset(shuffleB, DUMMY, sizeof(int) * BLOCK_DATA_SIZE);
      Msize2 = std::min(BLOCK_DATA_SIZE, inSize - (total_blocks-1-blocks_done) * BLOCK_DATA_SIZE);
      opOneLinearScanBlock((total_blocks-1-blocks_done) * BLOCK_DATA_SIZE, shuffleB, Msize2, inStructureId, 0, 0);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, shuffleB, BLOCK_DATA_SIZE, inStructureId, 1, 0);
      blocks_done += 1;
      if (blocks_done == total_blocks) {
        break;
      }
    }
    int blockNum = moveDummy(trustedM3, dataBoundary);
    quickSortMulti(trustedM3, 0, blockNum-1, samples, 1, p0, partitionIdx);
    sort(partitionIdx.begin(), partitionIdx.end());
    partitionIdx.insert(partitionIdx.begin(), -1);
    for (int j = 0; j < p0; ++j) {
      index1 = partitionIdx[j]+1;
      index2 = partitionIdx[j+1];
      // std::cout << trustedM3[index1] << ' ' << trustedM3[index2] << std::endl;
      writeBackNum = index2 - index1 + 1;
      if (writeBackNum == 0) {
        continue;
      }
      // Find out #elements=pivots=samples[j]
      equalNum = 0;
      eachNum = 0;
      // TODO: partition trustedM3[index1]
      index3 = partitionEqual(&trustedM3[index1], writeBackNum, samples[j]);
      if (j != 0 && index3 != -1) {
        // std::cout << trustedM3[index1+index3] << std::endl;
        
        index3 += index1 + 1;
        equalNum = index3 - index1;
        int parts = dupPivots[samples[j]];
        int average = equalNum / parts;
        int remainder = equalNum % parts;
        int startJ = BSFirstGreaterEqual(samples, samples[j]);
        int readM3Idx = 0;
        for (int m = 0; m < parts; ++m) {
          eachNum = average + ((m < remainder) ? 1 : 0);
          if (eachNum > smallSectionSize) {
            std::cout << "Overflow in small section1 M/p0: " << eachNum << std::endl;
          }
          opOneLinearScanBlock((startJ+m)*bucketSize0+i*smallSectionSize, &trustedM3[index1+readM3Idx], eachNum, outStructureId1, 1, smallSectionSize-eachNum);
          readM3Idx += eachNum;
        }
      }
      // TODO: Change reading start
      if (eachNum+writeBackNum-equalNum > smallSectionSize) {
        std::cout << "Overflow in small section2 M/p0: " << eachNum+writeBackNum-equalNum << std::endl;
      }
      opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize + eachNum, &trustedM3[index1+equalNum], writeBackNum-equalNum, outStructureId1, 1, smallSectionSize-eachNum-(writeBackNum-equalNum));
    }
    memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
    partitionIdx.clear();
  }
  free(trustedM3);
  free(shuffleB);
  if (bucketSize0 > M) {
    std::cout << "Each section size is greater than M, adjst parameters: " << bucketSize0 << ", " << M << std::endl;
  }
  return {bucketSize0, p0};
}*/

std::pair<int, int> OneLevelPartition(int inStructureId, int inSize, std::vector<int> &samples, int sampleSize, int p, int outStructureId1, int is_rec=0, int is_duplicate=0) {
  if (inSize <= M) {
    return {inSize, 1};
  }
  double beta = (!is_rec) ? BETA : _BETA;
  int hatN = ceil(1.0 * (1 + 2 * beta) * inSize);
  int M_prime = ceil(1.0 * M / (1 + 2 * beta));
  int r = ceil(1.0 * log(hatN / M) / log(p));
  int p0 = ceil(1.0 * hatN / (M * pow(p, r - 1)));
  quantileCal(samples, 0, sampleSize, p0);
  int boundary1 = ceil(1.0 * inSize / M_prime);
  int boundary2 = ceil(1.0 * M_prime / BLOCK_DATA_SIZE);
  int dataBoundary = boundary2 * BLOCK_DATA_SIZE;
  int smallSectionSize = M / p0;
  int bucketSize0 = boundary1 * smallSectionSize;
  freeAllocate(outStructureId1, outStructureId1, boundary1 * smallSectionSize * p0);
  
  int k, Msize1, Msize2, index1, index2, writeBackNum;
  int blocks_done = 0;
  int total_blocks = ceil(1.0 * inSize / BLOCK_DATA_SIZE);
  int *trustedM3 = (int*)malloc(sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  int *shuffleB = (int*)malloc(sizeof(int) * BLOCK_DATA_SIZE);
  std::vector<int> partitionIdx;
  // TODO: Find FFSEM implementation in c++
  for (int i = 0; i < boundary1; ++i) {
    for (int j = 0; j < boundary2; ++j) {
      if (total_blocks - 1 - blocks_done == 0) {
        k = 0;
      } else {
        k = rand() % (total_blocks - blocks_done);
      }
      Msize1 = std::min(BLOCK_DATA_SIZE, inSize - k * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, &trustedM3[j*BLOCK_DATA_SIZE], Msize1, inStructureId, 0, 0);
      memset(shuffleB, DUMMY, sizeof(int) * BLOCK_DATA_SIZE);
      Msize2 = std::min(BLOCK_DATA_SIZE, inSize - (total_blocks-1-blocks_done) * BLOCK_DATA_SIZE);
      opOneLinearScanBlock((total_blocks-1-blocks_done) * BLOCK_DATA_SIZE, shuffleB, Msize2, inStructureId, 0, 0);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, shuffleB, BLOCK_DATA_SIZE, inStructureId, 1, 0);
      blocks_done += 1;
      if (blocks_done == total_blocks) {
        break;
      }
    }
    int blockNum = moveDummy(trustedM3, dataBoundary);
    quickSortMulti(trustedM3, 0, blockNum-1, samples, 1, p0, partitionIdx);
    sort(partitionIdx.begin(), partitionIdx.end());
    partitionIdx.insert(partitionIdx.begin(), -1);
    for (int j = 0; j < p0; ++j) {
      index1 = partitionIdx[j]+1;
      index2 = partitionIdx[j+1];
      writeBackNum = index2 - index1 + 1;
      if (writeBackNum > smallSectionSize) {
        std::cout << "Overflow in small section M/p0: " << writeBackNum << ',' << smallSectionSize <<std::endl;
      }
      opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, &trustedM3[index1], writeBackNum, outStructureId1, 1, smallSectionSize - writeBackNum);
    }
    memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
    partitionIdx.clear();
  }
  free(trustedM3);
  free(shuffleB);
  if (bucketSize0 > M) {
    std::cout << "Each section size is greater than M, adjst parameters: " << bucketSize0 << ", " << M << std::endl;
  }
  return {bucketSize0, p0};
}



// TODO: Add TwoLevelPartition
std::pair<int, int> TwoLevelPartition(int inStructureId, std::vector<std::vector<int>>& pivots, int p, int outStructureId1, int outStructureId2) {
  int M_prime = ceil(1.0 * M / (1 + 2 * BETA));
  int p0 = p;
  int boundary1 = ceil(1.0 * N / M_prime);
  int boundary2 = ceil(1.0 * M_prime / BLOCK_DATA_SIZE);
  int dataBoundary = boundary2 * BLOCK_DATA_SIZE;
  int smallSectionSize = M / p0;
  int bucketSize0 = boundary1 * smallSectionSize;
  freeAllocate(outStructureId1, outStructureId1, boundary1 * smallSectionSize * p0);
  int k, Msize1, Msize2, index1, index2, writeBackNum;
  int blocks_done = 0;
  int total_blocks = ceil(1.0 * N / BLOCK_DATA_SIZE);
  int *trustedM3 = (int*)malloc(sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
  int *shuffleB = (int*)malloc(sizeof(int) * BLOCK_DATA_SIZE);
  std::vector<int> partitionIdx;
  for (int i = 0; i < boundary1; ++i) {
    for (int j = 0; j < boundary2; ++j) {
      if (total_blocks - 1 - blocks_done == 0) {
        k = 0;
      } else {
        k = rand() % (total_blocks - blocks_done);
      }
      Msize1 = std::min(BLOCK_DATA_SIZE, N - k * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, &trustedM3[j*BLOCK_DATA_SIZE], Msize1, inStructureId, 0, 0);
      memset(shuffleB, DUMMY, sizeof(int) * BLOCK_DATA_SIZE);
      Msize2 = std::min(BLOCK_DATA_SIZE, N - (total_blocks-1-blocks_done) * BLOCK_DATA_SIZE);
      opOneLinearScanBlock((total_blocks-1-blocks_done) * BLOCK_DATA_SIZE, shuffleB, Msize2, inStructureId, 0, 0);
      opOneLinearScanBlock(k * BLOCK_DATA_SIZE, shuffleB, BLOCK_DATA_SIZE, inStructureId, 1, 0);
      blocks_done += 1;
      if (blocks_done == total_blocks) {
        break;
      }
    }
    int blockNum = moveDummy(trustedM3, dataBoundary);
    quickSortMulti(trustedM3, 0, blockNum-1, pivots[0], 1, p0, partitionIdx);
    sort(partitionIdx.begin(), partitionIdx.end());
    partitionIdx.insert(partitionIdx.begin(), -1);
    for (int j = 0; j < p0; ++j) {
      index1 = partitionIdx[j]+1;
      index2 = partitionIdx[j+1];
      writeBackNum = index2 - index1 + 1;
      if (writeBackNum > smallSectionSize) {
        std::cout << "Overflow in small section M/p0: " << writeBackNum << std::endl;
      }
      opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, &trustedM3[index1], writeBackNum, outStructureId1, 1, smallSectionSize - writeBackNum);
    }
    memset(trustedM3, DUMMY, sizeof(int) * boundary2 * BLOCK_DATA_SIZE);
    partitionIdx.clear();
  }
  free(trustedM3);
  free(shuffleB);
  // Level2
  int p1 = p0 * p, readSize, readSize2, k1, k2;
  int boundary3 = ceil(1.0 * bucketSize0 / M);
  int bucketSize1 = boundary3 * smallSectionSize;
  freeAllocate(outStructureId2, outStructureId2, boundary3 * smallSectionSize * p0 * p);
  // std::vector<int> trustedM1;
  int *trustedM2 = (int*)malloc(sizeof(int) * M);
  // TODO: Change memory excceeds? use &trustedM2[M-1-p]
  int *trustedM2_part = (int*)malloc(sizeof(int) * (p+1));
  for (int j = 0; j < p0; ++j) {
    // trustedM1 = pivots[1+j];
    for (int k = 0; k < boundary3; ++k) {
      Msize1 = std::min(M, bucketSize0 - k * M);
      readSize = (Msize1 < (p+1)) ? Msize1 : (Msize1-(p+1));
      opOneLinearScanBlock(j*bucketSize0+k*M, trustedM2, readSize, outStructureId1, 0, 0);
      k1 = moveDummy(trustedM2, readSize);
      readSize2 = (Msize1 < (p+1)) ? 0 : (p+1);
      opOneLinearScanBlock(j*bucketSize0+k*M+readSize, trustedM2_part, readSize2, outStructureId1, 0, 0);
      k2 = moveDummy(trustedM2_part, readSize2);
      memcpy(&trustedM2[k1], trustedM2_part, sizeof(int) * k2);
      quickSortMulti(trustedM2, 0, k1+k2-1, pivots[1+j], 1, p, partitionIdx);
      sort(partitionIdx.begin(), partitionIdx.end());
      partitionIdx.insert(partitionIdx.begin(), -1);
      for (int ll = 0; ll < p; ++ll) {
        index1 = partitionIdx[ll]+1;
        index2 = partitionIdx[ll+1];
        writeBackNum = index2 - index1 + 1;
        if (writeBackNum > smallSectionSize) {
          std::cout << "Overflow in small section M/p0: " << writeBackNum << std::endl;
        }
        opOneLinearScanBlock((j*p0+ll)*bucketSize1+k*smallSectionSize, &trustedM2[index1], writeBackNum, outStructureId2, 1, smallSectionSize-writeBackNum);
      }
      memset(trustedM2, DUMMY, sizeof(int) * M);
      partitionIdx.clear();
    }
  }
  if (bucketSize1 > M) {
    std::cout << "Each section size is greater than M, adjust parameters: " << bucketSize1 << std::endl;
  }
  return {bucketSize1, p1};
}


int ObliviousTightSort(int inStructureId, int inSize, int outStructureId1, int outStructureId2) {
  int *trustedM;
  std::cout << "In ObliviousTightSort\n";
  if (inSize <= M) {
    trustedM = (int*)malloc(sizeof(int) * M);
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0, 0);
    quickSort(trustedM, 0, inSize - 1);
    freeAllocate(outStructureId1, outStructureId1, inSize);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1, 0);
    free(trustedM);
    return outStructureId1;
  }
  std::vector<int> trustedM2;
  int realNum = Sample(inStructureId, inSize, trustedM2, is_tight);
  std::pair<int, int> section= OneLevelPartition(inStructureId, inSize, trustedM2, realNum, P, outStructureId1);
  int sectionSize = section.first;
  int sectionNum = section.second;
  // TODO: IN order to reduce memory, can replace outStructureId2 with inStructureId
  freeAllocate(outStructureId2, outStructureId2, inSize);
  trustedM = (int*)malloc(sizeof(int) * M);
  int j = 0, k;
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k-1);
    opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1, 0);
    j += k;
    if (j > inSize) {
      std::cout << "Final error" << std::endl;
    }
  }
  free(trustedM);
  return outStructureId2;
}

// TODO: TightSort2
int ObliviousTightSort2(int inStructureId, int inSize, int sampleId, int sortedSampleId, int outStructureId1, int outStructureId2) {
  std::cout << "In ObliviousTightSort2 && In SampleRec\n";
  std::vector<std::vector<int>> pivots;
  SampleRec(inStructureId, sampleId, sortedSampleId, 1, pivots);
  std::cout << "In TwoLevelPartition\n";
  std::pair<int, int> section = TwoLevelPartition(inStructureId, pivots, P, outStructureId1, outStructureId2);
  std::cout << "Till Partition IOcost: " << IOcost/N*BLOCK_DATA_SIZE << std::endl;
  int sectionSize = section.first;
  int sectionNum = section.second;
  int *trustedM = (int*)malloc(sizeof(int) * M);
  int j = 0, k;
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 0, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k-1);
    opOneLinearScanBlock(j, trustedM, k, outStructureId1, 1, 0);
    j += k;
    if (j > inSize) {
      std::cout << "Final error2\n";
    }
  }
  std::cout << "Till Final IOcost: " << IOcost/N*BLOCK_DATA_SIZE << std::endl;
  return outStructureId1;
}

std::pair<int, int> ObliviousLooseSort(int inStructureId, int inSize, int outStructureId1, int outStructureId2) {
  std::cout << "In ObliviousLooseSort\n";
  int *trustedM;
  if (inSize <= M) {
    trustedM = (int*)malloc(sizeof(int) * M);
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0, 0);
    quickSort(trustedM, 0, inSize - 1);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1, 0);
    free(trustedM);
    return {outStructureId1, inSize};
  }
  std::vector<int> trustedM2;
  int realNum = Sample(inStructureId, inSize, trustedM2, is_tight);
  std::pair<int, int> section = OneLevelPartition(inStructureId, inSize, trustedM2, realNum, P, outStructureId1);
  int sectionSize = section.first;
  int sectionNum = section.second;
  int totalLevelSize = sectionNum * sectionSize;
  int k;
  freeAllocate(outStructureId2, outStructureId2, totalLevelSize);
  trustedM = (int*)malloc(sizeof(int) * M);
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1, 0);
  }
  return {outStructureId2, totalLevelSize};
}

// TODO: Finish
std::pair<int, int> ObliviousLooseSort2(int inStructureId, int inSize, int sampleId, int sortedSampleId, int outStructureId1, int outStructureId2) {
  std::cout << "In ObliviousLooseSort2 && In SasmpleRec\n";
  std::vector<std::vector<int>> pivots;
  SampleRec(inStructureId, sampleId, sortedSampleId, 0, pivots);
  std::cout << "Till Sample IOcost: " << IOcost/N*BLOCK_DATA_SIZE << std::endl;
  std::cout << "In TwoLevelPartition\n";
  std::pair<int, int> section = TwoLevelPartition(inStructureId, pivots, P, outStructureId1, outStructureId2);
  int sectionSize = section.first;
  int sectionNum = section.second;
  int totalLevelSize = sectionNum * sectionSize;
  int k;
  freeAllocate(outStructureId1, outStructureId1, totalLevelSize);
  int *trustedM = (int*)malloc(sizeof(int) * M);
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 0, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1, 0);
  }
  std::cout << "Till Final IOcost: " << IOcost/N*BLOCK_DATA_SIZE << std::endl;
  return {outStructureId1, totalLevelSize};
}

// TODO: Finish, what's return value
void ObliviousLooseSortRec(int sampleId, int sampleSize, int sortedSampleId, std::vector<std::vector<int>>& pivots) {
  std::cout << "In ObliviousLooseSortRec\n";
  std::vector<int> trustedM2;
  int realNum = Sample(sampleId, sampleSize, trustedM2, 0, 1);
  std::cout << "In OneLevelPartition\n";
  std::pair<int, int> section = OneLevelPartition(sampleId, sampleSize, trustedM2, realNum, _P, sortedSampleId, 1, 0);
  int sectionSize = section.first;
  int sectionNum = section.second;
  int j = 0, k = 0, total = 0;
  int outj = 0, inj = 0;
  int *trustedM = (int*)malloc(sizeof(int) * M);
  std::vector<int> quantileIdx;
  for (int i = 1; i < P; ++i) {
    quantileIdx.push_back(i * sampleSize / P);
  }
  int size = ceil(1.0 * sampleSize / P);
  std::vector<std::vector<int>> quantileIdx2;
  std::vector<int> index;
  for (int i = 0; i < P; ++i) {
    for (int j = 1; j < P; ++j) {
      index.push_back(i * size + j * size / P);
    }
    quantileIdx2.push_back(index);
    index.clear();
  }
  std::vector<int> pivots1, pivots2_part;
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, sortedSampleId, 0, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k-1);
    total += k;
    // Cal Level1 pivots
    while ((j < P-1) && (quantileIdx[j] < total)) {
      pivots1.push_back(trustedM[quantileIdx[j]-(total-k)]);
      j += 1;
    }
    // Cal Level2 pivots
    while (outj < P) {
      while ((inj < P-1) && (quantileIdx2[outj][inj] < total)) {
        pivots2_part.push_back(trustedM[quantileIdx2[outj][inj]-(total-k)]);
        inj += 1;
        if (inj == P-1) {
          inj = 0;
          outj += 1;
          pivots.push_back(pivots2_part);
          pivots2_part.clear();
          break;
        }
      }
      if (outj == P || quantileIdx2[outj][inj] >= total) {
        break;
      }
    }
    if ((j >= P-1) && (outj >= P)) {
      break;
    }
  }
  pivots1.insert(pivots1.begin(), INT_MIN);
  pivots1.push_back(INT_MAX);
  for (int i = 0; i < P; ++i) {
    pivots[i].insert(pivots[i].begin(), INT_MIN);
    pivots[i].push_back(INT_MAX);
  }
  pivots.insert(pivots.begin(), pivots1);
}


/*****************Auxiliary FUnctions****************/
// TODO: Remove?
// generate random numbers using uniform distribution

/*
mbedtls_aes_context aes;

void prf(unsigned char *right, char key, int tweak, unsigned char *ret) {
  unsigned char input[16] = {0};
  unsigned char encrypt_output[16] = {0};
  input[0] = right[0];
  input[1] = right[1];
  input[15] = tweak & 0xFF;
  mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_ENCRYPT, input, encrypt_output);
  ret[0] = encrypt_output[0];
  ret[1] = encrypt_output[1];
}

void round(unsigned char* data, char key, int tweak, unsigned char* newData) {
  unsigned char leftBits[2] = {data[0], data[1]};
  unsigned char rightBits[2] = {data[2], data[3]};
  newData[0] = rightBits[0];
  newData[1] = rightBits[1];
  unsigned char prfRet[2];
  prf(rightBits, key, tweak, prfRet);
  newData[2] = leftBits[0] ^ prfRet[0];
  newData[3] = leftBits[1] ^ prfRet[1];
}

// char key[8] = "=-÷×&";
int encrypt(int index, char key[8], int rounds) {
  unsigned char bytes[4];
  for (int i = 0; i < 4; ++i) {
    bytes[i] = (index >> (24 - i * 8)) & 0xFF;
  }
  int keyIdx = 0;
  unsigned char newBytes[4];
  while (rounds--) {
    round(bytes, key[keyIdx], rounds, newBytes);
    keyIdx = (keyIdx + 1) % 8;
  }
  int newIndex = 0;
  for (int i = 0; i < 4; ++i) {
    newIndex |= newBytes[i] << (24 - i * 8);
  }
  return newIndex;
}
*/

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

// TODO: Later change to vector, using internal sort
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
// calculating params ? how ?



// trusted function
void callSort(int sortId, int structureId, int paddedSize, int *resId, int *resN) {
  // TODO: Utilize Memory alloction -- structureId
  if (sortId == 0) {
    is_tight = 1;
    if (paddedSize / M < 100) {
      *resId = ObliviousTightSort(structureId, paddedSize, structureId + 1, structureId);
    } else {
      *resId = ObliviousTightSort2(structureId, paddedSize, structureId+1, structureId+2, structureId+1, structureId);
    }
  } else if (sortId == 1) {
    if (paddedSize / M < 100) {
      is_tight = 0;
      std::pair<int, int> ans = ObliviousLooseSort(structureId, paddedSize, structureId + 1, structureId);
      *resId = ans.first;
      *resN = ans.second;
    } else {
      std::pair<int, int> ans = ObliviousLooseSort2(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 1, structureId);
      *resId = ans.first;
      *resN = ans.second;
    }
  }
}


/** procedure test() : verify sort results **/
void init(int **arrayAddr, int structurId, int size) {
  int i;
  int *addr = (int*)arrayAddr[structurId];
  for (i = 0; i < size; i++) {
    addr[i] = (size - i);
    // addr[i] = (size - i) % 100;
    /*
    if (i < 1000000) {
      addr[i] = 10;
    } else if (i < 2000000) {
      addr[i] = 10;
    } else if (i < 3000000) {
      addr[i] = 10;
    } else if (i < 4000000) {
      addr[i] = 10;
    } else if (i < 5000000) {
      addr[i] = 10;
    } else if (i < 6000000) {
      addr[i] = 10;
    } else if (i < 7000000) {
      addr[i] = 4;
    } else if (i < 8000000) {
      addr[i] = 4;
    } else if (i < 9000000) {
      addr[i] = 4;
    } else if (i < 10000000) {
      addr[i] = 4;
    }*/
  }
  /*
  for(i = size - 1; i >= 1; --i) {
    swapRow(addr + i, addr + (rand() % i));
  }*/
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
  std::ofstream fout("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/OQSORT/OQSORT/output.txt");
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
      pass &= ((arrayAddr[structureId])[i-1] <= (arrayAddr[structureId])[i]);
      if (!pass) {
        std::cout << (arrayAddr[structureId])[i-1] << ' ' << (arrayAddr[structureId])[i];
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



