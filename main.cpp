//
//  main.cpp
//  ALLSORT
//
//  Created by ChampionNan on 10/5/2022.
//

//
//  main.c
//  bitonic
//
//  Created by ChampionNan on 28/4/2022.
//


//bitonic.c
/*
 This file contains two different implementations of the bitonic sort
        recursive  version
        imperative version :  impBitonicSort()
 

 The bitonic sort is also known as Batcher Sort.
 For a reference of the algorithm, see the article titled
 Sorting networks and their applications by K. E. Batcher in 1968


 The following codes take references to the codes avaiable at

 http://www.cag.lcs.mit.edu/streamit/results/bitonic/code/c/bitonic.c

 http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

 http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 */

/*
------- ----------------------
   Nikos Pitsianis, Duke CS
-----------------------------
*/

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

#define N 9437184//10000000
#define M 1048576 // int type memory restriction
#define NUM_STRUCTURES 10
// #define MEM_IN_ENCLAVE 5
#define DUMMY 0xffffffff
#define NULLCHAR '\0'
// #define B 10

#define ALPHA 0.02
#define BETA 0.093
#define P 11

#define FAN_OUT 2
#define BLOCK_DATA_SIZE 4
#define BUCKET_SIZE 337//256
#define MERGE_BATCH_SIZE 20 // merge split hepler
#define HEAP_NODE_SIZE 20//8192. heap node size
#define WRITE_BUFFER_SIZE 20



typedef struct {
  int x;
  int key;
} Bucket_x;

bool cmpHelper(Bucket_x *a, Bucket_x *b) {
  return (a->x > b->x) ? true : false;
}

bool cmpHelper(int *a, int *b) {
  return (*a > *b) ? true : false;
}

struct HeapNode {
  Bucket_x *data;
  int bucketIdx;
  int elemIdx;
};


class Heap {
  HeapNode *harr;
  int heapSize;
  int batchSize;
public:
  Heap(HeapNode *a, int size, int bsize);
  void Heapify(int i);
  int left(int i);
  int right (int i);
  void swapHeapNode(HeapNode *a, HeapNode *b);
  HeapNode *getRoot();
  int getHeapSize();
  bool reduceSizeByOne();
  void replaceRoot(HeapNode x);
};



Heap::Heap(HeapNode *a, int size, int bsize) {
  heapSize = size;
  harr = a;
  int i = (heapSize - 1) / 2;
  batchSize = bsize;
  while (i >= 0) {
    Heapify(i);
    i --;
  }
}

void Heap::Heapify(int i) {
  int l = left(i);
  int r = right(i);
  int target = i;

  if (l < heapSize && cmpHelper(harr[i].data + harr[i].elemIdx % batchSize, harr[l].data + harr[l].elemIdx % batchSize)) {
    target = l;
  }
  if (r < heapSize && cmpHelper(harr[target].data + harr[target].elemIdx % batchSize, harr[r].data + harr[r].elemIdx % batchSize)) {
    target = r;
  }
  if (target != i) {
    swapHeapNode(&harr[i], &harr[target]);
    Heapify(target);
  }
}

int Heap::left(int i) {
  return (2 * i + 1);
}

int Heap::right(int i) {
  return (2 * i + 2);
}

void Heap::swapHeapNode(HeapNode *a, HeapNode *b) {
  HeapNode temp = *a;
  *a = *b;
  *b = temp;
}

HeapNode* Heap::getRoot() {
  return &harr[0];
}

int Heap::getHeapSize() {
  return heapSize;
}

bool Heap::reduceSizeByOne() {
  free(harr[0].data);
  heapSize --;
  if (heapSize > 0) {
    harr[0] = harr[heapSize];
    Heapify(0);
    return true;
  } else {
    return false;
  }
}

void Heap::replaceRoot(HeapNode x) {
  harr[0] = x;
  Heapify(0);
}

int printf(const char *fmt, ...);
int greatestPowerOfTwoLessThan(int n);
int smallestPowerOfKLargerThan(int n, int k);
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write);
void padWithDummy(int structureId, int start, int realNum, int secSize);
bool isTargetIterK(int randomKey, int iter, int k, int num);
void swapRow(Bucket_x *a, Bucket_x *b);
void swapRow(int *a, int *b);
void init(int **arrayAddr, int structurId, int size);
void print(int* array, int size);
void print(int **arrayAddr, int structureId, int size);
void test(int **arrayAddr, int structureId, int size);
void testWithDummy(int **arrayAddr, int structureId, int size);

void callSort(int sortId, int structureId, int paddedSize, int *resId, int *resN);
void smallBitonicMerge(int *a, int start, int size, int flipped);
void smallBitonicSort(int *a, int start, int size, int flipped);
void bitonicMerge(int structureId, int start, int size, int flipped, int* row1, int* row2);
void bitonicSort(int structureId, int start, int size, int flipped, int* row1, int* row2);
int greatestPowerOfTwoLessThan(int n);
void mergeSplitHelper(Bucket_x *inputBuffer, int* numRow1, int* numRow2, int* inputId, int* outputId, int iter, int k, int* bucketAddr, int outputStructureId);
void mergeSplit(int inputStructureId, int outputStructureId, int *inputId, int *outputId, int k, int* bucketAddr, int* numRow1, int* numRow2, int iter);
void kWayMergeSort(int inputStructureId, int outputStructureId, int* numRow1, int* bucketAddr, int bucketNum);
void bucketSort(int inputStructureId, int bucketId, int size, int dataStart);
int bucketOSort(int structureId, int size);
int partition(Bucket_x *arr, int low, int high);
int partition(int *arr, int low, int high);
void quickSort(Bucket_x *arr, int low, int high);
void quickSort(int *arr, int low, int high);
int moveDummy(int *a, int size);

int Hypergeometric(int NN, int Msize, double n_prime);


int *X;
//structureId=1, bucket1 in bucket sort; input
Bucket_x *bucketx1;
//structureId=2, bucket 2 in bucket sort
Bucket_x *bucketx2;
//structureId=3, write back array
int *Y;
int *arrayAddr[NUM_STRUCTURES];
int paddedSize;
// TODO: set up structure size
const int structureSize[NUM_STRUCTURES] = {sizeof(int),
  sizeof(Bucket_x), sizeof(Bucket_x),
  sizeof(int), sizeof(int), sizeof(Bucket_x), sizeof(Bucket_x), sizeof(int), sizeof(int)};


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
}

void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId) {
  if (blockSize == 0) {
    // printf("Unknown data size");
    return;
  }
  // memcpy(arrayAddr[structureId] + index, buffer, blockSize * structureSize[structureId]);
  memcpy(arrayAddr[structureId] + index, buffer, blockSize);
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
  
  // 0: OSORT-Tight, 1: OSORT-Loss, 2: bucketOSort, 3: bitonicSort
  int sortId = 1;
  
  int inputId = 0;

  // step1: init test numbers
  if (sortId == 3) {
    // inputId = 0;
    int addi = 0;
    if (N % BLOCK_DATA_SIZE != 0) {
      addi = ((N / BLOCK_DATA_SIZE) + 1) * BLOCK_DATA_SIZE - N;
    }
    X = (int*)malloc((N + addi) * sizeof(int));
    paddedSize = N + addi;
    arrayAddr[inputId] = X;
    init(arrayAddr, inputId, paddedSize);
  } else if (sortId == 2) {
    // inputId = 0;
    assert(FAN_OUT >= 2 && "M/Z must greater than 2");
    int bucketNum = smallestPowerOfKLargerThan(ceil(2.0 * N / BUCKET_SIZE), FAN_OUT);
    int bucketSize = bucketNum * BUCKET_SIZE;
    std::cout << "TOTAL BUCKET SIZE: " << bucketSize << std::endl;
    std::cout << "BUCKET NUMBER: " << bucketNum << std::endl;
    bucketx1 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
    bucketx2 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
    memset(bucketx1, 0xff, bucketSize*sizeof(Bucket_x));
    memset(bucketx2, 0xff, bucketSize*sizeof(Bucket_x));
    arrayAddr[1] = (int*)bucketx1;
    arrayAddr[2] = (int*)bucketx2;
    X = (int *) malloc(N * sizeof(int));
    arrayAddr[inputId] = X;
    paddedSize = N;
    init(arrayAddr, inputId, paddedSize);
  } else if (sortId == 0 || sortId == 1) {
    inputId = 3;
    X = (int *)malloc(N * sizeof(int));
    arrayAddr[inputId] = X;
    paddedSize = N;
    init(arrayAddr, inputId, paddedSize);
  }

  // step2: Create the enclave
  // print(arrayAddr, inputId, N);
  
  // step3: call sort algorithms
  start = std::chrono::high_resolution_clock::now();
  if (sortId == 3) {
    std::cout << "Test bitonic sort... " << std::endl;
    callSort(sortId, inputId, paddedSize, resId,  resN);
    test(arrayAddr, inputId, paddedSize);
  } else if (sortId == 2) {
    std::cout << "Test bucket oblivious sort... " << std::endl;
    callSort(sortId, inputId + 1, paddedSize, resId, resN);
    std::cout << "Result ID: " << *resId << std::endl;
    print(arrayAddr, *resId, N);
    test(arrayAddr, *resId, paddedSize);
  } else if (sortId == 0 || sortId == 1) {
    std::cout << "Test OQSort... " << std::endl;
    callSort(sortId, inputId, paddedSize, resId, resN);
    std::cout << "Result ID: " << *resId << std::endl;
    if (sortId == 0) {
      test(arrayAddr, *resId, paddedSize);
    } else {
      // Sample Loose has different test & print
      testWithDummy(arrayAddr, *resId, *resId);
    }
    print(arrayAddr, *resId, *resN);
  }
  end = std::chrono::high_resolution_clock::now();
  

  // step4: std::cout execution time
  duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Finished. Duration Time: " << duration.count() << " seconds" << std::endl;

  // step5: exix part
  exit:
    
    for (int i = 0; i < NUM_STRUCTURES; ++i) {
      if (arrayAddr[i]) {
        // TODO: i=5?error
        free(arrayAddr[i]);
      }
    }
    free(resId);
    free(resN);
    return ret;
}

int greatestPowerOfTwoLessThan(int n) {
    int k = 1;
    while (k > 0 && k < n) {
        k = k << 1;
    }
    return k >> 1;
}

int smallestPowerOfKLargerThan(int n, int k) {
  int num = 1;
  while (num > 0 && num < n) {
    num = num * k;
  }
  return num;
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

// Combnatortial number
int CombiNum(int n, int m) {
  if (n == m || m == 0) {
    return 1;
  }
  std::vector<int> dp(m + 1);
  for (int i = 0; i <= n; i++) {
    for (int j = std::min(i, m); j >= 0; j--) {
      if (i == j || j == 0) {
        dp[j] = 1;
      } else {
        dp[j] = dp[j] + dp[j - 1];
      }
    }
  }
  return dp[m];
}

// TODO: calculate Hypergeometric Distribution
int Hypergeometric(int NN, int Msize, double n_prime) {
  int m = 0;
  std::random_device rd;
  std::mt19937_64 generator(rd());
  double rate = n_prime / double(NN);
  std::bernoulli_distribution b(rate);
  for (int j = 0; j < Msize; ++j) {
    if (b(generator)) {
      m ++;
      n_prime -= 1;
    }
    NN -= 1;
    rate = n_prime / double(NN);
    std::bernoulli_distribution b(rate);
  }
  return m;
}
  
void shuffle(int *array, int n) {
  if (n > 1) {
    for (int i = 0; i < n - 1; ++i) {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int SampleTight(int inStructureId, int samplesId) {
  int N_prime = N;
  double n_prime = 1.0 * ALPHA * N;
  int alphaM2 = (int)ceil(2.0 * ALPHA * M);
  int boundary = (int)ceil(1.0 * N/M);
  int Msize, alphaM22;
  int m; // use for hypergeometric distribution
  int realNum = 0; // #pivots
  int writeBackstart = 0;
  int readStart = 0;
  int *trustedMemory = (int*)malloc(M * sizeof(int));
  
  freeAllocate(samplesId, samplesId, alphaM2 * boundary);
  
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(M, N - i * M);
    alphaM22 = (int)ceil(2.0 * ALPHA * Msize);
    opOneLinearScanBlock(readStart, trustedMemory, Msize, inStructureId, 0);
    // print(trustedMemory, Msize);
    readStart += Msize;
    // step1. sample with hypergeometric distribution
    m = Hypergeometric(N_prime, Msize, n_prime);
    if (m > alphaM22) {
      return -1;
    }
    realNum += m;
    // step2. shuffle M
    shuffle(trustedMemory, Msize);
    // step3. set dummy
    memset(&trustedMemory[m], DUMMY, (Msize - m) * sizeof(int));
    // step4. write sample back to external memory
    opOneLinearScanBlock(writeBackstart, trustedMemory, alphaM22, samplesId, 1);
    writeBackstart += alphaM22;
    N_prime -= Msize;
    n_prime -= m;
  }
  
  if (realNum < M) {
    opOneLinearScanBlock(0, trustedMemory, writeBackstart, samplesId, 0);
    int realN = moveDummy(trustedMemory, writeBackstart);
    if (realN != realNum) {
      std::cout << "Counting error after moving dummy.\n";
    }
    double nonDummyNum = ALPHA * N;
    std::cout << realN << ", " << nonDummyNum << std::endl;
    quickSort(trustedMemory, 0, realN - 1);
    opOneLinearScanBlock(0, trustedMemory, realN, samplesId, 1);
    // print(arrayAddr, samplesId, realN);
  } else {
    std::cout << "RealNum >= M\n";
    return -1;
  }
  free(trustedMemory);
  return realNum;
}

int SampleLoose(int inStructureId, int samplesId) {
  int N_prime = N;
  double n_prime = 1.0 * ALPHA * N;
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
      opOneLinearScanBlock(k, trustedMemory, m, samplesId, 1);
      k += m;
    }
    N_prime -= Msize;
    n_prime -= m;
    if (n_prime <= 0) {
      break;
    }
    // TODO: ? what's m value
  }
  trustedMemory = (int*)realloc(trustedMemory, M * sizeof(int));
  if (realNum < M) {
    opOneLinearScanBlock(0, trustedMemory, realNum, samplesId, 0);
    quickSort(trustedMemory, 0, realNum - 1);
    opOneLinearScanBlock(0, trustedMemory, realNum, samplesId, 1);
  } else {
    std::cout << "RealNum >= M\n";
    return -1;
  }
  std::cout << "RealNum:" << realNum << std::endl;
  // print(arrayAddr, samplesId, realNum);
  free(trustedMemory);
  return realNum;
}

int upperBound(int *a, int size, int k) {
  int start = 0;
  int last = size;
  while (start < last) {
    int mid = start + (last - start) / 2;
    if (a[mid] <= k) {
      start = mid + 1;
    } else {
      last = mid;
    }
  }
  return start;
}

// TODO: here sample size = 2 * n ? [start, end)
int quantileCal(int sampleId, int start, int end, int p, int *trustedM1) {
  int *pivotIdx = (int*)malloc(sizeof(int) * (p+1));
  // vector<int>
  int sampleSize = end - start;
  for (int i = 1; i < p; i ++) {
    pivotIdx[i] = i * sampleSize / p;
  }
  int *trustedMemory = (int*)malloc(sizeof(int) * BLOCK_DATA_SIZE);
  int boundary = (int)ceil(1.0 * sampleSize / BLOCK_DATA_SIZE);
  int Msize;
  int totalRead = 0;
  int j = 1; // record pivotId
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(BLOCK_DATA_SIZE, sampleSize - i * BLOCK_DATA_SIZE);
    opOneLinearScanBlock(start + i * BLOCK_DATA_SIZE, trustedMemory, Msize, sampleId, 0);
    totalRead += Msize;
    while (pivotIdx[j] < totalRead) {
      trustedM1[j] = trustedMemory[pivotIdx[j] % BLOCK_DATA_SIZE];
      j++;
      if (j == p) {
        break;
      }
    }
    if (j == p) {
      break;
    }
  }
  free(pivotIdx);
  free(trustedMemory);
  trustedM1[0] = INT_MIN;
  trustedM1[p] = INT_MAX;
  // TODO: ADD test pivots correct value
  bool passFlag = 1;
  for (int i = 0; i < p; ++i) {
    std::cout << trustedM1[i] << ", " << trustedM1[i+1] << std::endl;
    passFlag &= (trustedM1[i] < trustedM1[i+1]);
    if (trustedM1[i + 1] < 0) {
      passFlag = 0;
      break;
    }
  }
  return passFlag;
}

int ProcessL(int LIdIn, int LIdOut, int lsize) {
  // TODO: bucket type
  freeAllocate(LIdIn, LIdIn, lsize * 2);
  freeAllocate(LIdOut, LIdOut, lsize * 2);
  Bucket_x *L = (Bucket_x*)malloc(sizeof(Bucket_x) * BLOCK_DATA_SIZE);
  int Msize;
  int boundary = (int)ceil(1.0 * lsize / BLOCK_DATA_SIZE);
  int k = 0;
  // 1. Initialize array L and set up random Key
  for (int i = 0; i < boundary; ++i) {
    Msize = std::min(BLOCK_DATA_SIZE, lsize - i * BLOCK_DATA_SIZE);
    opOneLinearScanBlock(2 * i * BLOCK_DATA_SIZE, (int*)L, Msize, LIdIn, 0);
    for (int i = 0; i < Msize; ++i) {
      // L[i].x = (int)oe_rand();
      L[i].x = (int)rand();
      L[i].key = k++;
    }
    opOneLinearScanBlock(2 * i * BLOCK_DATA_SIZE, (int*)L, Msize, LIdIn, 1);
  }
  
  // TODO: External Memory Sort: eg merge sort
  int bucketNum = (int)ceil(1.0 * lsize / M);
  int bucketSize = lsize / bucketNum;
  int residual = lsize % bucketNum;
  int totalSize = 0;
  int *numRow = (int*)malloc(sizeof(int) * bucketNum);
  int *bucketAddr = (int*)malloc(sizeof(int) * bucketNum);
  memset(numRow, 0, sizeof(int) * bucketNum);
  for (int i = 0; i < bucketNum; ++i) {
    numRow[i] = bucketSize + (i < residual ? 1 : 0);
    if (i == 0) {
      bucketAddr[i] = 0;
    } else {
      totalSize += numRow[i - 1];
      bucketAddr[i] = totalSize;
    }
  }
  for (int i = 0; i < bucketNum; ++i) {
    std::cout << LIdIn << ", " << i << ", " <<numRow[i] << ", " << bucketAddr[i] << std::endl;
    bucketSort(LIdIn, i, numRow[i], bucketAddr[i]);
  }
  // print(arrayAddr, LIdIn, lsize);
  kWayMergeSort(LIdIn, LIdOut, numRow, bucketAddr, bucketNum);
  std::cout << "Test Process L order after kwaymergesort\n";
  test(arrayAddr, LIdOut, lsize);
  free(numRow);
  free(bucketAddr);
  free(L);
  return 0;
}

// inStructureId: original input array
// outStructureId1&2: intermidiate level data storage
std::pair<int, int> MultiLevelPartition(int inStructureId, int sampleId, int LIdIn, int LIdOut, int sampleSize, int p, int outStructureId1) {
  if (N <= M) {
    return std::make_pair(N, 1);
  }
  int hatN = (int)ceil(1.0 * (1 + 2 * BETA) * N);
  int M_prime = (int)ceil(1.0 * M / (1 + 2 * BETA));
  // 1. Initialize array L, extrenal memory
  int lsize = (int)ceil(1.0 * N / BLOCK_DATA_SIZE);
  // 2. set up block index array L & shuffle L
  ProcessL(LIdIn, LIdOut, lsize);
  // freeAllocate(LIdIn, LIdIn, 0);
  
  int r = (int)ceil(1.0 * log(hatN / M) / log(p));
  int p0 = (int)ceil(1.0 * hatN / (M * pow(p, r - 1)));
  
  // 3. calculate p0-quantile about sample
  int *trustedM1 = (int*)malloc(sizeof(int) * (p0 + 1));
  bool resFlag = quantileCal(sampleId, 0, sampleSize, p0, trustedM1);
  while (!resFlag) {
    std::cout << "Quantile calculate error!\n";
    print(arrayAddr, sampleId, sampleSize);
    resFlag = quantileCal(sampleId, 0, sampleSize, p0, trustedM1);
  }
  print(trustedM1, p0 + 1);
  // print(arrayAddr, sampleId, sampleSize);
  // 4. allocate trusted memory
  int boundary1 = (int)ceil(2.0 * N / M_prime);
  int boundary2 = (int)ceil(1.0 * M_prime / (2 * BLOCK_DATA_SIZE));
  Bucket_x *trustedM2 = (Bucket_x*)malloc(sizeof(Bucket_x) * boundary2);
  int *trustedM3 = (int*)malloc(sizeof(int) * ((int)ceil(1.0 * M_prime / 2)));
  
  int **bucketNum = (int**)malloc(sizeof(int*) * p0);
  for (int i = 0; i < p0; ++i) {
    bucketNum[i] = (int*)malloc(sizeof(int) * boundary1);
    memset(bucketNum[i], 0, sizeof(int) * boundary1);
  }
  int Msize1, Msize2;
  int smallSectionSize = (int)ceil(1.0 * M / (2 * p0));
  // 5. allocate out memory using for index
  freeAllocate(outStructureId1, outStructureId1, boundary1 * smallSectionSize * p0);
  int bucketSize0 = boundary1 * smallSectionSize;
  
  int size1 = (int)ceil(1.0 * N / BLOCK_DATA_SIZE);
  int M3size;
  int blockIdx;
  // 6. level0 partition
  for (int i = 0; i < boundary1; ++i) {
    Msize1 = std::min(boundary2, size1 - i * boundary2);
    if (Msize1 <= 0) {
      // TODO: ?
      break;
    }
    opOneLinearScanBlock(2 * i * boundary2, (int*)trustedM2, Msize1, LIdOut, 0);
    M3size = 0; // all input number
    for (int j = 0; j < Msize1; ++j) {
      Msize2 = std::min(BLOCK_DATA_SIZE, N - j * BLOCK_DATA_SIZE);
      blockIdx = trustedM2[j].key;
      opOneLinearScanBlock(blockIdx * BLOCK_DATA_SIZE, &trustedM3[j * BLOCK_DATA_SIZE], Msize2, inStructureId, 0);
      M3size += Msize2;
    }
    for (int j = 0; j < p0; ++j) {
      // std::cout << "Pivots: " << trustedM1[j] << ", " << trustedM1[j + 1] << std::endl;
      for (int m = 0; m < M3size; ++m) {
        // iterate M3 && M3 contains
        // std::cout << "Data: " << trustedM3[m] << std::endl;
        if ((trustedM3[m] != DUMMY) && (trustedM3[m] >= trustedM1[j]) && (trustedM3[m] < trustedM1[j + 1])) {
          opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize + bucketNum[j][i], &trustedM3[m], 1, outStructureId1, 1);
          bucketNum[j][i] += 1;
          if (bucketNum[j][i] > smallSectionSize) {
            std::cout << "Overflow in small section M/2p0: " << bucketNum[j][i] << std::endl;
          }
        }
      }
    }
  }
  // 7. pad dummy
  // std::cout << "Before padding: \n";
  // print(arrayAddr, outStructureId1, boundary1 * smallSectionSize * p0);
  for (int j = 0; j < p0; ++j) {
    for (int i = 0; i < boundary1; ++i) {
      padWithDummy(outStructureId1, j * bucketSize0 + i * smallSectionSize, bucketNum[j][i], smallSectionSize);
    }
  }
  // std::cout << "After padding: \n";
  // print(arrayAddr, outStructureId1, boundary1 * smallSectionSize * p0);
  free(trustedM1);
  free(trustedM2);
  free(trustedM3);
  for (int i = 0; i < p0; ++i) {
    free(bucketNum[i]);
  }
  free(bucketNum);
  if (bucketSize0 > M) {
    std::cout << "Each section size is greater than M, adjst parameters.\n";
  }
  // freeAllocate(LIdOut, LIdOut, 0);
  return std::make_pair(bucketSize0, p0);
}


int ObliviousTightSort(int inStructureId, int inSize, int sampleId, int LIdIn, int LIdOut, int outStructureId1, int outStructureId2) {
  int *trustedM = (int*)malloc(sizeof(int) * M);
  if (inSize <= M) {
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0);
    quickSort(trustedM, 0, inSize - 1);
    freeAllocate(outStructureId1, outStructureId1, inSize);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1);
    return outStructureId1;
  }
  int realNum = SampleTight(inStructureId, sampleId);
  int n = (int)ceil(1.0 * ALPHA * N);
  while (realNum < 0) {
    std::cout << "Samples number error!\n";
    realNum = SampleTight(inStructureId, sampleId);
  }
  
  std::pair<int, int> section = MultiLevelPartition(inStructureId, sampleId, LIdIn, LIdOut, std::min(realNum, n), P, outStructureId1);
  int sectionSize = section.first;
  int sectionNum = section.second;
  int totalLevelSize = sectionNum * sectionSize;
  int j = 0;
  int k;
  
  freeAllocate(outStructureId2, outStructureId2, inSize);
  
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0);
    // TODO: optimize to utilize bucketNum[j][i]
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1);
    j += k;
    if (j > inSize) {
      std::cout << "Overflow" << std::endl;
    }
  }
  std::cout << "After inter-sorting: \n";
  // print(arrayAddr, outStructureId2, inSize);
  return outStructureId2;
}

std::pair<int, int> ObliviousLooseSort(int inStructureId, int inSize, int sampleId, int LIdIn, int LIdOut, int outStructureId1, int outStructureId2) {
  int *trustedM = (int*)malloc(sizeof(int) * M);
  if (inSize <= M) {
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0);
    quickSort(trustedM, 0, inSize - 1);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1);
    return {outStructureId1, inSize};
  }
  int realNum = SampleLoose(inStructureId, sampleId);
  int n = (int)ceil(1.0 * ALPHA * N);
  while (realNum < 0) {
    std::cout << "Samples number error!\n";
    realNum = SampleLoose(inStructureId, sampleId);
  }

  std::pair<int, int> section = MultiLevelPartition(inStructureId, sampleId, LIdIn, LIdOut, std::min(realNum, n), P, outStructureId1);
  int sectionSize = section.first;
  int sectionNum = section.second;
  int totalLevelSize = sectionNum * sectionSize;
  int j = 0;
  int k, Msize;
  
  freeAllocate(outStructureId2, outStructureId2, totalLevelSize);
  
  for (int i = 0; i < sectionNum; ++i) {
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0);
    k = moveDummy(trustedM, sectionSize);
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1);
  }
  
  /*
  for (int i = 0; i < sectionNum; ++i) {
    Msize = std::min(sectionSize, totalLevelSize - i * sectionSize);
    opOneLinearScanBlock(i * sectionSize, trustedM, Msize, outStructureId1, 0);
    // TODO: optimize to utilize bucketNum[j][i]
    k = moveDummy(trustedM, Msize);
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1);
    j += k;
    if (j > inSize) {
      std::cout << "Overflow" << std::endl;
    }
  }*/
  
  return {outStructureId2, totalLevelSize};
}


// bitonic sort
void smallBitonicMerge(int *a, int start, int size, int flipped) {
  if (size == 1) {
    return;
  } else {
    int swap = 0;
    int mid = greatestPowerOfTwoLessThan(size);
    for (int i = 0; i < size - mid; ++i) {
      int num1 = a[start + i];
      int num2 = a[start + mid + i];
      swap = num1 > num2;
      swap = swap ^ flipped;
      a[start + i] = (!swap * num1) + (swap * num2);
      a[start + i + mid] = (swap * num1) + (!swap * num2);
    }
    smallBitonicMerge(a, start, mid, flipped);
    smallBitonicMerge(a, start + mid, size - mid, flipped);
  }
}

void smallBitonicSort(int *a, int start, int size, int flipped) {
  if (size <= 1) {
    return;
  } else {
    int mid = greatestPowerOfTwoLessThan(size);
    smallBitonicSort(a, start, mid, 1);
    smallBitonicSort(a, start + mid, size - mid, 0);
    smallBitonicMerge(a, start, size, flipped);
  }
}

void bitonicMerge(int structureId, int start, int size, int flipped, int* row1, int* row2) {
  if (size < 1) {
    return ;
  } else if (size * BLOCK_DATA_SIZE < M) {
    int *trustedMemory = (int*)malloc(size * BLOCK_DATA_SIZE * structureSize[structureId]);
    for (int i = 0; i < size; ++i) {
      opOneLinearScanBlock((start + i) * BLOCK_DATA_SIZE, &trustedMemory[i * BLOCK_DATA_SIZE], BLOCK_DATA_SIZE, structureId, 0);
    }
    smallBitonicMerge(trustedMemory, 0, size * BLOCK_DATA_SIZE, flipped);
    for (int i = 0; i < size; ++i) {
      opOneLinearScanBlock((start + i) * BLOCK_DATA_SIZE, &trustedMemory[i * BLOCK_DATA_SIZE], BLOCK_DATA_SIZE, structureId, 1);
    }
    free(trustedMemory);
  } else {
    int swap = 0;
    int mid = greatestPowerOfTwoLessThan(size);
    for (int i = 0; i < size - mid; ++i) {
      opOneLinearScanBlock((start + i) * BLOCK_DATA_SIZE, row1, BLOCK_DATA_SIZE, structureId, 0);
      opOneLinearScanBlock((start + mid + i) * BLOCK_DATA_SIZE, row2, BLOCK_DATA_SIZE, structureId, 0);
      int num1 = row1[0], num2 = row2[0];
      swap = num1 > num2;
      swap = swap ^ flipped;
      for (int j = 0; j < BLOCK_DATA_SIZE; ++j) {
        int v1 = row1[j];
        int v2 = row2[j];
        row1[j] = (!swap * v1) + (swap * v2);
        row2[j] = (swap * v1) + (!swap * v2);
      }
      opOneLinearScanBlock((start + i) * BLOCK_DATA_SIZE, row1, BLOCK_DATA_SIZE, structureId, 1);
      opOneLinearScanBlock((start + mid + i) * BLOCK_DATA_SIZE, row2, BLOCK_DATA_SIZE, structureId, 1);
    }
    bitonicMerge(structureId, start, mid, flipped, row1, row2);
    bitonicMerge(structureId, start + mid, size - mid, flipped, row1, row2);
  }
  return;
}

void bitonicSort(int structureId, int start, int size, int flipped, int* row1, int* row2) {
  if (size < 1) {
    return;
  } else if (size * BLOCK_DATA_SIZE < M) {
    int *trustedMemory = (int*)malloc(size * BLOCK_DATA_SIZE * structureSize[structureId]);
    for (int i = 0; i < size; ++i) {
      opOneLinearScanBlock((start + i) * BLOCK_DATA_SIZE, &trustedMemory[i * BLOCK_DATA_SIZE], BLOCK_DATA_SIZE, structureId, 0);
    }
    smallBitonicSort(trustedMemory, 0, size * BLOCK_DATA_SIZE, flipped);
    // write back
    for (int i = 0; i < size; ++i) {
      opOneLinearScanBlock((start + i) * BLOCK_DATA_SIZE, &trustedMemory[i * BLOCK_DATA_SIZE], BLOCK_DATA_SIZE, structureId, 1);
    }
    free(trustedMemory);
  } else {
    int mid = greatestPowerOfTwoLessThan(size);
    bitonicSort(structureId, start, mid, 1, row1, row2);
    bitonicSort(structureId, start + mid, size - mid, 0, row1, row2);
    bitonicMerge(structureId, start, size, flipped, row1, row2);
  }
  return;
}

// trusted function
void callSort(int sortId, int structureId, int paddedSize, int *resId, int *resN) {
  if (sortId == 0) {
    *resId = ObliviousTightSort(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 3, structureId + 4, structureId + 5);
  } else if (sortId == 1) {
    std::pair<int, int> ans = ObliviousLooseSort(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 3, structureId + 4, structureId + 5);
    *resId = ans.first;
    *resN = ans.second;
  } else if (sortId == 2) {
     *resId = bucketOSort(structureId, paddedSize);
  } else if (sortId == 3) {
    int size = paddedSize / BLOCK_DATA_SIZE;
    int *row1 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
    int *row2 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
    bitonicSort(structureId, 0, size, 0, row1, row2);
    free(row1);
    free(row2);
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
  
  } else if (structureSize[structureId] == 8) {
    Bucket_x *junk = (Bucket_x*)malloc(len * sizeof(Bucket_x));
    for (int i = 0; i < len; ++i) {
      junk[i].x = DUMMY;
      junk[i].key = DUMMY;
    }
    opOneLinearScanBlock(2 * (start + realNum), (int*)junk, len, structureId, 1);
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

bool isTargetIterK(int randomKey, int iter, int k, int num) {
  while (iter) {
    randomKey = randomKey / k;
    iter--;
  }
  // return (randomKey & (0x01 << (iter - 1))) == 0 ? false : true;
  return (randomKey % k) == num;
}

void mergeSplitHelper(Bucket_x *inputBuffer, int* numRow1, int* numRow2, int* inputId, int* outputId, int iter, int k, int* bucketAddr, int outputStructureId) {
  // int batchSize = BUCKET_SIZE; // 8192
  // TODO: FREE these malloc
  Bucket_x **buf = (Bucket_x**)malloc(k * sizeof(Bucket_x*));
  for (int i = 0; i < k; ++i) {
    buf[i] = (Bucket_x*)malloc(BUCKET_SIZE * sizeof(Bucket_x));
  }
  
  // int counter0 = 0, counter1 = 0;
  int randomKey;
  int *counter = (int*)malloc(k * sizeof(int));
  memset(counter, 0, k * sizeof(int));
  
  for (int i = 0; i < k * BUCKET_SIZE; ++i) {
    if ((inputBuffer[i].key != DUMMY) && (inputBuffer[i].x != DUMMY)) {
      randomKey = inputBuffer[i].key;
      for (int j = 0; j < k; ++j) {
        if (isTargetIterK(randomKey, iter, k, j)) {
          buf[j][counter[j] % BUCKET_SIZE] = inputBuffer[i];
          counter[j]++;
          // std::cout << "couter j: " << counter[j] << std::endl;
          if (counter[j] % BUCKET_SIZE == 0) {
            opOneLinearScanBlock(2 * (bucketAddr[outputId[j]] +  numRow2[outputId[j]]), (int*)buf[j], (size_t)BUCKET_SIZE, outputStructureId, 1);
            numRow2[outputId[j]] += BUCKET_SIZE;
          }
        }
      }
    }
  }
  
  for (int j = 0; j < k; ++j) {
    opOneLinearScanBlock(2 * (bucketAddr[outputId[j]] + numRow2[outputId[j]]), (int*)buf[j], (size_t)(counter[j] % BUCKET_SIZE), outputStructureId, 1);
    numRow2[outputId[j]] += counter[j] % BUCKET_SIZE;
    padWithDummy(outputStructureId, bucketAddr[outputId[j]], numRow2[outputId[j]], BUCKET_SIZE);
    if (numRow2[outputId[j]] > BUCKET_SIZE) {
      printf("overflow error during merge split!\n");
    }
    free(buf[j]);
  }
  free(counter);
}

void mergeSplit(int inputStructureId, int outputStructureId, int *inputId, int *outputId, int k, int* bucketAddr, int* numRow1, int* numRow2, int iter) {
  // step1. Read k buckets together
  Bucket_x *inputBuffer = (Bucket_x*)malloc(k * sizeof(Bucket_x) * BUCKET_SIZE);
  for (int i = 0; i < k; ++i) {
    opOneLinearScanBlock(2 * bucketAddr[inputId[i]], (int*)(&inputBuffer[i * BUCKET_SIZE]), BUCKET_SIZE, inputStructureId, 0);
  }
  // step2. process k buckets
  mergeSplitHelper(inputBuffer, numRow1, numRow2, inputId, outputId, iter, k, bucketAddr, outputStructureId);
  free(inputBuffer);
  
  
}

void kWayMergeSort(int inputStructureId, int outputStructureId, int* numRow1, int* bucketAddr, int bucketNum) {
  int mergeSortBatchSize = HEAP_NODE_SIZE; // 256
  int writeBufferSize = (int)WRITE_BUFFER_SIZE; // 8192
  int numWays = bucketNum;
  // HeapNode inputHeapNodeArr[numWays];
  HeapNode *inputHeapNodeArr = (HeapNode*)malloc(numWays * sizeof(HeapNode));
  int totalCounter = 0;
  
  int *readBucketAddr = (int*)malloc(sizeof(int) * numWays);
  memcpy(readBucketAddr, bucketAddr, sizeof(int) * numWays);
  int writeBucketAddr = 0;
  int j = 0;
  
  for (int i = 0; i < numWays; ++i) {
    // TODO: 数据0跳过
    if (numRow1[i] == 0) {
      continue;
    }
    HeapNode node;
    node.data = (Bucket_x*)malloc(mergeSortBatchSize * sizeof(Bucket_x));
    node.bucketIdx = i;
    node.elemIdx = 0;
    opOneLinearScanBlock(2 * readBucketAddr[i], (int*)node.data, (size_t)std::min(mergeSortBatchSize, numRow1[i]), inputStructureId, 0);
    inputHeapNodeArr[j++] = node;
    readBucketAddr[i] += std::min(mergeSortBatchSize, numRow1[i]);
  }
  
  Heap heap(inputHeapNodeArr, j, mergeSortBatchSize);
  Bucket_x *writeBuffer = (Bucket_x*)malloc(writeBufferSize * sizeof(Bucket_x));
  int writeBufferCounter = 0;

  while (1) {
    HeapNode *temp = heap.getRoot();
    memcpy(writeBuffer + writeBufferCounter, temp->data + temp->elemIdx % mergeSortBatchSize, sizeof(Bucket_x));
    writeBufferCounter ++;
    totalCounter ++;
    temp->elemIdx ++;
    
    if (writeBufferCounter == writeBufferSize) {
      opOneLinearScanBlock(2 * writeBucketAddr, (int*)writeBuffer, (size_t)writeBufferSize, outputStructureId, 1);
      writeBucketAddr += writeBufferSize;
      // numRow2[temp->bucketIdx] += writeBufferSize;
      writeBufferCounter = 0;
      // print(arrayAddr, outputStructureId, numWays * BUCKET_SIZE);
    }
    
    if (temp->elemIdx < numRow1[temp->bucketIdx] && (temp->elemIdx % mergeSortBatchSize) == 0) {
      opOneLinearScanBlock(2 * readBucketAddr[temp->bucketIdx], (int*)(temp->data), (size_t)std::min(mergeSortBatchSize, numRow1[temp->bucketIdx]-temp->elemIdx), inputStructureId, 0);
      
      readBucketAddr[temp->bucketIdx] += std::min(mergeSortBatchSize, numRow1[temp->bucketIdx]-temp->elemIdx);
      heap.Heapify(0);
      
    } else if (temp->elemIdx >= numRow1[temp->bucketIdx]) {
      bool res = heap.reduceSizeByOne();
      if (!res) {
        break;
      }
    } else {
      heap.Heapify(0);
    }
  }
  opOneLinearScanBlock(2 * writeBucketAddr, (int*)writeBuffer, (size_t)writeBufferCounter, outputStructureId, 1);
  // numRow2[0] += writeBufferCounter;
  // TODO: ERROR writeBuffer
  free(writeBuffer);
  free(readBucketAddr);
  free(inputHeapNodeArr);
}

void bucketSort(int inputStructureId, int bucketId, int size, int dataStart) {
  Bucket_x *arr = (Bucket_x*)malloc(size * sizeof(Bucket_x));
  opOneLinearScanBlock(2 * dataStart, (int*)arr, (size_t)size, inputStructureId, 0);
  quickSort(arr, 0, size - 1);
  opOneLinearScanBlock(2 * dataStart, (int*)arr, (size_t)size, inputStructureId, 1);
  free(arr);
}

// int inputTrustMemory[BLOCK_DATA_SIZE];
int bucketOSort(int structureId, int size) {
  int k = FAN_OUT;
  int bucketNum = smallestPowerOfKLargerThan(ceil(2.0 * size / BUCKET_SIZE), k);
  int mem1 = 2 * 2 * k * BUCKET_SIZE + bucketNum * 3;
  int mem2 = 3 * bucketNum + bucketNum * HEAP_NODE_SIZE * 2 + 2 * WRITE_BUFFER_SIZE;
  if ((mem1 > M) || (mem2 > M)) {
    std::cout << "Mem1's memory " << mem1 << " * 4 bytes exceeds\n";
    std::cout << "Mem2's memory " << mem2 << " * 4 bytes exceeds\n";
  }
  int ranBinAssignIters = log(bucketNum)/log(k) - 1;
  std::cout << "Iteration times: " << log(bucketNum)/log(k) << std::endl;
  srand((unsigned)time(NULL));
  // std::cout << "Iters:" << ranBinAssignIters << std::endl;
  int *bucketAddr = (int*)malloc(bucketNum * sizeof(int));
  for (int i = 0; i < bucketNum; ++i) {
    bucketAddr[i] = i * BUCKET_SIZE;
  }
  int *numRow1 = (int*)malloc(bucketNum * sizeof(int));
  memset(numRow1, 0, bucketNum * sizeof(int));
  int *numRow2 = (int*)malloc(bucketNum * sizeof(int));
  memset(numRow2, 0, bucketNum * sizeof(int));
  
  
  Bucket_x *trustedMemory = (Bucket_x*)malloc(BLOCK_DATA_SIZE * sizeof(Bucket_x));
  int *inputTrustMemory = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  int total = 0;
  int offset;

  for (int i = 0; i < size; i += BLOCK_DATA_SIZE) {
    opOneLinearScanBlock(i, inputTrustMemory, std::min(BLOCK_DATA_SIZE, size - i), structureId - 1, 0);
    int randomKey;
    for (int j = 0; j < std::min(BLOCK_DATA_SIZE, size - i); ++j) {
      // randomKey = (int)oe_rdrand();
      randomKey = rand();
      trustedMemory[j].x = inputTrustMemory[j];
      trustedMemory[j].key = randomKey;
      
      offset = bucketAddr[(i + j) % bucketNum] + numRow1[(i + j) % bucketNum];
      opOneLinearScanBlock(offset * 2, (int*)(&trustedMemory[j]), (size_t)1, structureId, 1);
      numRow1[(i + j) % bucketNum] ++;
    }
    total += std::min(BLOCK_DATA_SIZE, size - i);
  }
  free(trustedMemory);
  free(inputTrustMemory);

  for (int i = 0; i < bucketNum; ++i) {
    //printf("currently bucket %d has %d records/%d\n", i, numRow1[i], BUCKET_SIZE);
    padWithDummy(structureId, bucketAddr[i], numRow1[i], BUCKET_SIZE);
  }
  // print(arrayAddr, structureId, bucketNum * BUCKET_SIZE);
  // std::cout << "Iters:" << ranBinAssignIters << std::endl;
  // std::cout << "k:" << k << std::endl;
  int *inputId = (int*)malloc(k * sizeof(int));
  int *outputId = (int*)malloc(k *sizeof(int));
  int outIdx = 0;
  
  for (int i = 0; i < ranBinAssignIters; ++i) {
    if (i % 2 == 0) {
      for (int j = 0; j < bucketNum / (int)pow(k, i+1); ++j) {
        // pass (i-1) * k^i
        //printf("j: %d\n", j);
        for (int jj = 0; jj < (int)pow(k, i); ++jj) {
          //printf("jj: %d\n", jj);
          for (int m = 0; m < k; ++m) {
            //printf("j, jj, m: %d, %d, %d\n", j, jj, m);
            inputId[m] = j * (int)pow(k, i+1)+ jj + m * (int)pow(k, i);
            outputId[m] = (outIdx * k + m) % bucketNum;
            //printf("input, output: %d, %d\n", inputId[m], outputId[m]);
          }
          mergeSplit(structureId, structureId + 1, inputId, outputId, k, bucketAddr, numRow1, numRow2, i);
          outIdx ++;
        }
      }
      int count = 0;
      for (int n = 0; n < bucketNum; ++n) {
        numRow1[n] = 0;
        count += numRow2[n];
      }
      printf("after %dth merge split, we have %d tuples\n", i, count);
      outIdx = 0;
      //print(arrayAddr, structureId + 1, bucketNum * BUCKET_SIZE);
    } else {
      for (int j = 0; j < bucketNum / (int)pow(k, i+1); ++j) {
        //printf("j: %d\n", j);
        for (int jj = 0; jj < (int)pow(k, i); ++jj) {
          //printf("jj: %d\n", jj);
          for (int m = 0; m < k; ++m) {
            //printf("j, jj, m: %d, %d, %d\n", j, jj, m);
            inputId[m] = j * (int)pow(k, i+1)+ jj + m * (int)pow(k, i);
            outputId[m] = (outIdx * k + m) % bucketNum;
            //printf("input, output: %d, %d\n", inputId[m], outputId[m]);
          }
          mergeSplit(structureId + 1, structureId, inputId, outputId, k, bucketAddr, numRow2, numRow1, i);
          outIdx ++;
        }
      }
      int count = 0;
      for (int n = 0; n < bucketNum; ++n) {
        numRow2[n] = 0;
        count += numRow1[n];
      }
      printf("after %dth merge split, we have %d tuples\n", i, count);
      outIdx = 0;
      //print(arrayAddr, structureId, bucketNum * BUCKET_SIZE);
    }
    std::cout << "----------------------------------------\n";
    printf("\n\n Finish random bin assignment iter%dth out of %d\n\n", i, ranBinAssignIters);
    std::cout << "----------------------------------------\n";
  }
  free(inputId);
  free(outputId);
  int resultId = 0;
  if (ranBinAssignIters % 2 == 0) {
    for (int i = 0; i < bucketNum; ++i) {
      bucketSort(structureId, i, numRow1[i], bucketAddr[i]);
    }
    //std::cout << "********************************************\n";
    //print(arrayAddr, structureId, bucketNum * BUCKET_SIZE);
    
    kWayMergeSort(structureId, structureId + 1, numRow1, bucketAddr, bucketNum);
    
    resultId = structureId + 1;
  } else {
    for (int i = 0; i < bucketNum; ++i) {
      bucketSort(structureId + 1, i, numRow2[i], bucketAddr[i]);
    }
    //std::cout << "********************************************\n";
    //print(arrayAddr, structureId, bucketNum * BUCKET_SIZE);
    
    kWayMergeSort(structureId + 1, structureId, numRow2, bucketAddr, bucketNum);
    resultId = structureId;
  }
  // test(arrayAddr, resultId, N);
  // print(arrayAddr, resultId, N);
  free(bucketAddr);
  free(numRow1);
  free(numRow2);
  return resultId;
}

void swapRow(Bucket_x *a, Bucket_x *b) {
  Bucket_x *temp = (Bucket_x*)malloc(sizeof(Bucket_x));
  memmove(temp, a, sizeof(Bucket_x));
  memmove(a, b, sizeof(Bucket_x));
  memmove(b, temp, sizeof(Bucket_x));
  free(temp);
}

void swapRow(int *a, int *b) {
  int *temp = (int*)malloc(sizeof(int));
  memmove(temp, a, sizeof(int));
  memmove(a, b, sizeof(int));
  memmove(b, temp, sizeof(int));
  free(temp);
}


int partition(Bucket_x *arr, int low, int high) {
  int randNum = rand() % (high - low + 1) + low;
  swapRow(arr + high, arr + randNum);
  Bucket_x *pivot = arr + high;
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


void quickSort(Bucket_x *arr, int low, int high) {
  if (high > low) {
    int mid = partition(arr, low, high);
    quickSort(arr, low, mid - 1);
    quickSort(arr, mid + 1, high);
  }
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
  } else if (structureSize[structureId] == 8) {
    Bucket_x *addr = (Bucket_x*)arrayAddr[structureId];
    for (i = 0; i < size; i++) {
      // printf("(%d, %d) ", addr[i].x, addr[i].key);
      fout << "(" << addr[i].x << ", " << addr[i].key << ") ";
      if ((i != 0) && (i % 5 == 0)) {
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
      if ((arrayAddr[structureId])[i] == 0) {
        pass = 0;
        break;
      }
    }
  } else if (structureSize[structureId] == 8) {
    for (i = 1; i < size; i++) {
      pass &= (((Bucket_x*)arrayAddr[structureId])[i-1].x <= ((Bucket_x*)arrayAddr[structureId])[i].x);
      if (((Bucket_x*)arrayAddr[structureId])[i].x == 0) {
        pass = 0;
        break;
      }
    }
  }
  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
}

void testWithDummy(int **arrayAddr, int structureId, int size) {
  int i = 0;
  int j;
  // print(structureId);
  if(structureSize[structureId] == 4) {
    for (i = 0; i < size; ++i) {
      if ((arrayAddr[structureId])[i] != DUMMY) {
        break;
      }
    }
    if (i == size) { // All elements are dummy
      printf(" TEST PASSed\n");
      return;
    }
    while (i < size) {
      for (j = i + 1; j < size; ++i) {
        if ((arrayAddr[structureId])[j] != DUMMY) {
          break;
        }
      }
      if (j == size) { // Only 1 element not dummy
        printf(" TEST PASSed\n");
        return;
      }
      if ((arrayAddr[structureId])[i] <= (arrayAddr[structureId])[j]) {
        i = j;
      } else {
        printf(" TEST FAILed\n");
        return;
      }
    }
  } else if (structureSize[structureId] == 8) {
    for (i = 0; i < size; ++i) {
      if (((Bucket_x*)arrayAddr[structureId])[i].x != DUMMY) {
        break;
      }
    }
    if (i == size) { // All elements are dummy
      printf(" TEST PASSed\n");
      return;
    }
    while (i < size) {
      for (j = i + 1; j < size; ++i) {
        if (((Bucket_x*)arrayAddr[structureId])[j].x != DUMMY) {
          break;
        }
      }
      if (j == size) { // Only 1 element not dummy
        printf(" TEST PASSed\n");
        return;
      }
      if (((Bucket_x*)arrayAddr[structureId])[i].x <= ((Bucket_x*)arrayAddr[structureId])[j].x) {
        i = j;
      } else {
        printf(" TEST FAILed\n");
        return;
      }
    }
  }
}



