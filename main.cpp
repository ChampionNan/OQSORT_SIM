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

#define N 10000000//1100
#define M 80000 // int type memory restriction
#define NUM_STRUCTURES 10
// #define MEM_IN_ENCLAVE 5
#define DUMMY 0xffffffff
#define NULLCHAR '\0'
// #define B 10

#define ALPHA 0.1
#define BETA 0.1
#define GAMMA 0.1
#define P 5

#define FAN_OUT 20
#define BLOCK_DATA_SIZE 256
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
  return (a > b) ? true : false;
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
void padWithDummy(int structureId, int start, int realNum);
bool isTargetIterK(int randomKey, int iter, int k, int num);
void swapRow(Bucket_x *a, Bucket_x *b);
void init(int **arrayAddr, int structurId, int size);
void print(int* array, int size);
void print(int **arrayAddr, int structureId, int size);
void test(int **arrayAddr, int structureId, int size);
void callSort(int sortId, int structureId, int paddedSize, int *resId);
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
const int structureSize[NUM_STRUCTURES] = {sizeof(int), sizeof(Bucket_x), sizeof(Bucket_x)};


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
  // oe_result_t result;
  // oe_enclave_t* enclave = NULL;
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::seconds duration;
  // freopen("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/out.txt", "w+", stdout);
  // 0: OSORT-Tight, 1: OSORT-Loss, 2: bucketOSort, 3: bitonicSort
  int sortId = 2;
  int *oqId;

  // step1: init test numbers
  if (sortId == 3) {
    int addi = 0;
    if (N % BLOCK_DATA_SIZE != 0) {
      addi = ((N / BLOCK_DATA_SIZE) + 1) * BLOCK_DATA_SIZE - N;
    }
    X = (int*)malloc((N + addi) * sizeof(int));
    paddedSize = N + addi;
    arrayAddr[0] = X;
  } else if (sortId == 2) {
    srand((unsigned)time(NULL));
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
    arrayAddr[0] = X;
    paddedSize = N;
  } else if (sortId == 0 || sortId == 1) {
    oqId = (int*)malloc(sizeof(int) * 6);
    for (int i = 0; i < 6; ++i) {
      oqId[i] = i + 3;
    }
    X = (int *)malloc(N * sizeof(int));
    arrayAddr[oqId[0]] = X;
    paddedSize = N;
  }
  init(arrayAddr, 0, paddedSize);

  // step2: Create the enclave
  
  
  // step3: call sort algorithms
  start = std::chrono::high_resolution_clock::now();
  if (sortId == 3) {
    std::cout << "Test bitonic sort... " << std::endl;
    callSort(sortId, 0, paddedSize, resId);
    test(arrayAddr, 0, paddedSize);
  } else if (sortId == 2) {
    std::cout << "Test bucket oblivious sort... " << std::endl;
    callSort(sortId, 1, paddedSize, resId);
    std::cout << "Result ID: " << *resId << std::endl;
    // print(arrayAddr, *resId, N);
    test(arrayAddr, *resId, paddedSize);
  } else if (sortId == 0 || sortId == 1) {
    std::cout << "Test OQSort... " << std::endl;
    callSort(sortId, oqId[0], paddedSize, resId);
    std::cout << "Result ID: " << *resId << std::endl;
    // print(arrayAddr, *resId, N);
    test(arrayAddr, *resId, paddedSize);
  }
  end = std::chrono::high_resolution_clock::now();
  

  // step4: std::cout execution time
  duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Finished. Duration Time: " << duration.count() << " seconds" << std::endl;

  // step5: exix part
  exit:
    
    for (int i = 0; i < NUM_STRUCTURES; ++i) {
      if (arrayAddr[i]) {
        free(arrayAddr[i]);
      }
    }
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

// Functions x crossing the enclave boundary
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write) {
  if (!write) {
    OcallReadBlock(index, block, blockSize * structureSize[structureId], structureId);
    // OcallReadBlock(index, block, blockSize, structureId);
  } else {
    OcallWriteBlock(index, block, blockSize * structureSize[structureId], structureId);
    // OcallWriteBlock(index, block, blockSize, structureId);
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
int Hypergeometric(int NN, int Msize, int n_prime) {
  int m = 0;
  std::random_device rd;
  std::mt19937_64 generator(rd());
  double rate = ALPHA;
  std::bernoulli_distribution b(rate);
  for (int j = 0; j < Msize; ++j) {
    if (b(generator)) {
      m ++;
    }
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
  int n_prime = (int)ceil(ALPHA * N);
  int alphaM2 = (int)ceil(2 * ALPHA * M);
  int boundary = (int)ceil(N/M);
  int Msize;
  int m; // use for hypergeometric distribution
  int realNum = 0; // #pivots
  int writeBackstart = 0;
  int readStart = 0;
  int *trustedMemory = (int*)malloc(M * sizeof(int));
  
  freeAllocate(samplesId, samplesId, alphaM2 * boundary);
  
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(M, N - i * M);
    opOneLinearScanBlock(readStart, trustedMemory, Msize, inStructureId, 0);
    // print(trustedMemory, Msize);
    readStart += Msize;
    // step1. sample with hypergeometric distribution
    m = Hypergeometric(N_prime, M, n_prime);
    if (m > alphaM2) {
      return -1;
    }
    realNum += m;
    // step2. shuffle M
    shuffle(trustedMemory, Msize);
    // step3. set dummy
    memset(&trustedMemory[m], DUMMY, (Msize - m) * sizeof(int));
    // step4. write sample back to external memory
    opOneLinearScanBlock(writeBackstart, trustedMemory, alphaM2, samplesId, 1);
    writeBackstart += alphaM2;
    N_prime -= M;
    n_prime -= m;
  }
  
  if (realNum < M) {
    opOneLinearScanBlock(0, trustedMemory, realNum, samplesId, 0);
    quickSort(trustedMemory, 0, realNum - 1);
    opOneLinearScanBlock(0, trustedMemory, realNum, samplesId, 1);
  } else {
    // TODO: CALL ObliviousTightSort(); ?
  }
  free(trustedMemory);
  return realNum;
}

int SampleLoose(int inStructureId, int samplesId) {
  int N_prime = N;
  int n_prime = (int)ceil(ALPHA * N);
  int boundary = (int)ceil(N/BLOCK_DATA_SIZE);
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
    m = Hypergeometric(N_prime, BLOCK_DATA_SIZE, n_prime);
    if (m > 0) {
      realNum += std::min(m, n_prime);
      opOneLinearScanBlock(readStart, trustedMemory, Msize, inStructureId, 0);
      readStart += Msize;
      // step2. shuffle M
      shuffle(trustedMemory, Msize);
      // step4. write sample back to external memory
      opOneLinearScanBlock(k, trustedMemory, std::min(m, n_prime), samplesId, 1);
      k += m;
    }
    N_prime -= Msize;
    n_prime -= std::min(m, n_prime);
    if (n_prime <= 0) {
      break;
    }
    // TODO: ? what's m value
  }
  
  if (realNum < M) {
    opOneLinearScanBlock(0, trustedMemory, realNum, samplesId, 0);
    quickSort(trustedMemory, 0, realNum - 1);
    opOneLinearScanBlock(0, trustedMemory, realNum, samplesId, 1);
  } else {
    // TODO: CALL ObliviousTightSort(); ?
  }
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
  int *pivotIdx = (int*)malloc(sizeof(int) * (p + 1));
  int sampleSize = end - start;
  for (int i = 0; i < p; i ++) {
    pivotIdx[i] = i * sampleSize / p;
  }
  int *trustedMemory = (int*)malloc(sizeof(int) * BLOCK_DATA_SIZE);
  int boundary = (int)ceil(sampleSize / BLOCK_DATA_SIZE);
  int Msize;
  int totalRead = 0;
  int j = 0; // record pivotId
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(BLOCK_DATA_SIZE, sampleSize - i * BLOCK_DATA_SIZE);
    opOneLinearScanBlock(start + i * BLOCK_DATA_SIZE, trustedMemory, Msize, sampleId, 0);
    totalRead += Msize;
    while (pivotIdx[j] < totalRead) {
      trustedM1[j++] = trustedMemory[pivotIdx[j] % BLOCK_DATA_SIZE];
      if (j == p) {
        break;
      }
    }
  }
  free(pivotIdx);
  free(trustedMemory);
  return 0;
}

int ProcessL(int LIdIn, int LIdOut, int lsize) {
  // TODO: bucket type
  freeAllocate(LIdIn, LIdIn, lsize * 2);
  freeAllocate(LIdOut, LIdOut, lsize * 2);
  Bucket_x *L = (Bucket_x*)malloc(sizeof(Bucket_x) * BLOCK_DATA_SIZE);
  int Msize;
  int boundary = (int)ceil(lsize / BLOCK_DATA_SIZE);
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
  int bucketNum = (int)ceil(lsize / M);
  int bucketSize = lsize / bucketNum;
  int residual = bucketSize % bucketNum;
  int *numRow = (int*)malloc(sizeof(int) * bucketNum);
  int *bucketAddr = (int*)malloc(sizeof(int) * bucketNum);
  memset(numRow, 0, sizeof(int) * bucketNum);
  for (int i = 0; i < bucketNum; ++i) {
    numRow[i] = bucketSize + (i < residual ? 1 : 0);
    if (i == 0) {
      bucketAddr[i] = 0;
    } else {
      bucketAddr[i] = numRow[i - 1];
    }
  }
  for (int i = 0; i < bucketNum; ++i) {
    bucketSort(LIdIn, i, numRow[i], bucketAddr[i]);
  }
  kWayMergeSort(LIdIn, LIdOut, numRow, bucketAddr, bucketNum);
  free(numRow);
  free(bucketAddr);
  free(L);
  return 0;
}

// inStructureId: original input array
// outStructureId1&2: intermidiate level data storage
int MultiLevelPartition(int inStructureId, int sampleId, int LIdIn, int LIdOut, int sampleSize, int p, int outStructureId1, int outStructureId2) {
  if (N <= M) {
    return inStructureId;
  }
  int hatN = (int)ceil((1 + 2 * BETA) * N);
  int M_prime = (int)ceil(M / (1 + 2 * BETA));
  // 1. Initialize array L, extrenal memory
  int lsize = (int)ceil(N / BLOCK_DATA_SIZE);
  // 2. set up block index array L & shuffle L
  ProcessL(LIdIn, LIdOut, lsize);
  freeAllocate(LIdIn, LIdIn, 0);
  
  int r = (int)ceil(log(hatN / M) / log(p));
  int p0 = (int)ceil(hatN / (M * pow(p, r - 1)));
  // 2. Initialize array X
  int bucketSize0 = (int)ceil(hatN / p0);
  int bucketSize1;
  
  // 3. calculate p0-quantile about sample
  int *trustedM1 = (int*)malloc(sizeof(int) * (p0 + 1));
  quantileCal(sampleId, 0, sampleSize, p0, trustedM1);
  trustedM1[p0] = INT_MAX;
  // 4. allocate trusted memory
  int boundary1 = (int)ceil(2 * N / M_prime);
  int boundary2 = (int)ceil(M_prime / (2 * BLOCK_DATA_SIZE));
  int *trustedM2 = (int*)malloc(sizeof(int) * boundary2);
  int *trustedM3 = (int*)malloc(sizeof(int) * ((int)ceil(M_prime / 2)));
  
  int **bucketNum = (int**)malloc(sizeof(int*) * p0);
  for (int i = 0; i < p0; ++i) {
    bucketNum[i] = (int*)malloc(sizeof(int) * boundary1);
    memset(bucketNum[i], 0, sizeof(int) * boundary1);
  }
  int Msize1, Msize2;
  int smallSectionSize = (int)ceil(M / (2 * p0));
  // TODO: should change bsize to boundary1 * M/2p0
  freeAllocate(outStructureId1, outStructureId1, boundary1 * smallSectionSize * p0);
  int size1 = (int)ceil(N / BLOCK_DATA_SIZE);
  int M3size;
  // 5. level0 partition
  for (int i = 0; i < boundary1; ++i) {
    Msize1 = std::min(boundary2, size1 - i * boundary2);
    opOneLinearScanBlock(i * boundary2, trustedM2, Msize1, LIdOut, 0);
    M3size = 0; // all input number
    for (int j = 0; j <= Msize1; ++j) {
      Msize2 = std::min(BLOCK_DATA_SIZE, N - j * BLOCK_DATA_SIZE);
      opOneLinearScanBlock(j * BLOCK_DATA_SIZE, &trustedM3[j * BLOCK_DATA_SIZE], Msize2, inStructureId, 0);
      M3size += Msize2;
    }
    for (int j = 0; j < p0; ++j) {
      for (int m = 0; m < M3size; ++m) {
        // iterate M3 && M3 contains
        if (trustedM3[m] != DUMMY && trustedM3[m] >= trustedM1[j] && trustedM3[m] < trustedM1[j + 1]) {
          opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize + bucketNum[j][i], &trustedM3[m], 1, outStructureId1, 1);
          bucketNum[j][i] += 1;
          if (bucketNum[j][i] > smallSectionSize) {
            return -1; // Fail
          }
        }
      }
    }
  }
  // 6. pad dummy
  for (int j = 0; j < p0; ++j) {
    for (int i = 0; i < boundary1; ++i) {
      padWithDummy(outStructureId1, j * bucketSize0 + i * smallSectionSize, bucketNum[j][i]);
    }
  }
  free(trustedM1);
  free(trustedM2);
  free(trustedM3);
  // 7. later level partition
  trustedM1 = (int*)malloc(sizeof(int) * (p + 1));
  trustedM2 = (int*)malloc(sizeof(int) * ((int)ceil(M / 2)));
  int pi, pi_1, kboundary;
  smallSectionSize = (int)ceil(M / (2 * p));
  int outId1, outId2;
  for (int i = 1; i < r; i++) {
    if (i % 2 == 1) {
      outId1 = outStructureId1;
      outId2 = outStructureId2;
    } else {
      outId1 = outStructureId2;
      outId2 = outStructureId1;
    }
    if (i == 1) {
      pi_1 = p0;
      pi = p * p0;
    } else {
      pi_1 = pi;
      pi = p * pi;
    }
    for (int m = 0; m < pi_1; ++m) {
      free(bucketNum[m]);
    }
    free(bucketNum);
    bucketNum = (int**)malloc(sizeof(int*) * pi);
    kboundary = (int)ceil(2 * hatN / (pi_1 * M));
    // fix j, record each section number
    for (int n = 0; n < pi; ++n) {
      bucketNum[n] = (int*)malloc(sizeof(int) * kboundary);
      memset(bucketNum[n], 0, sizeof(int) * kboundary);
    }
    // TODO: FIX ?
    if (i != 1) {
      bucketSize0 = bucketSize1;
    }
    bucketSize1 = (int)ceil(hatN / pi);
    // change to for loop boundary multiply
    freeAllocate(outId2, outId2, p * pi_1 * kboundary * smallSectionSize);
    for (int j = 0; j < pi_1; ++j) {
      int pivotSec = (int)ceil(sampleSize / pi_1);
      quantileCal(sampleId, j * pivotSec, (j + 1) * pivotSec + 1, p, trustedM1);
      trustedM1[p] = INT_MAX;
      
      for (int k = 0; k <= kboundary; ++k) {
        Msize2 = std::min((int)ceil(M / 2), bucketSize0 - k * ((int)ceil(M / 2)));
        opOneLinearScanBlock(j * bucketSize0 + k * Msize2, trustedM2, Msize2, outId1, 0);
        for (int l = 0; l < p; ++l) {
          for (int m = 0; m < Msize2; ++m) {
            if (trustedM2[m] != DUMMY && trustedM2[m] >= trustedM1[l] && trustedM2[m] < trustedM1[l + 1]) {
              opOneLinearScanBlock((j * pi_1 + l) * bucketSize1 + k * smallSectionSize + bucketNum[j * pi_1 + l][k], &trustedM2[m], 1, outId2, 1);
              bucketNum[j * pi_1 + l][k] ++;
              if (bucketNum[j * pi_1 + l][k] > smallSectionSize) {
                return -1;
              }
            }
          }
        }
      }
    }
  }
  // 8. pad dummy
  for (int j = 0; j < pi; ++j) {
    for (int k = 0; k < kboundary; ++k) {
      padWithDummy(outId2, j * bucketSize1 + k * smallSectionSize, bucketNum[j][k]);
    }
  }
  // TODO: FREE all
  free(trustedM1);
  free(trustedM2);
  for (int i = 0; i < pi; ++i) {
    free(bucketNum[i]);
  }
  free(bucketNum);
  return pi;
}

// structureId=-1 -> use y in pivots selection use other sorting agorithm
// TODO: 需要使用MEM_IN_ENCLAVE来做enclave memory的约束? how
// TODO: 主要是用来限制oponelinear一次读取进来的数据大小

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
  int n = (int)ceil(ALPHA * N);
  while (realNum < 0 || realNum < n) {
    realNum = SampleTight(inStructureId, sampleId);
  }
  
  int sectionNum = MultiLevelPartition(inStructureId, sampleId, LIdIn, LIdOut, n, P, outStructureId1, outStructureId2);
  int j = 0;
  int k, Msize;
  int hatN = (int)ceil((1 + 2 * BETA) * N);
  int sectionSize = (int)ceil(hatN / sectionNum);
  int r = (int)ceil(log(hatN / M) / log(P));
  int multiOutId = (r % 2 == 1) ? outStructureId2 : outStructureId1;
  int resOutId = (r % 2 == 0) ? outStructureId2 : outStructureId1;
  
  for (int i = 0; i < sectionNum; ++i) {
    Msize = std::min(sectionSize, hatN - i * sectionSize);
    opOneLinearScanBlock(i * sectionSize, trustedM, Msize, multiOutId, 0);
    for (k = 0; k < Msize; ++k) {
      if (trustedM[k] == DUMMY) {
        break;
      }
    }
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(j, trustedM, k, resOutId, 1);
    j += k;
  }
  return resOutId;
}

int ObliviousLooseSort(int inStructureId, int inSize, int sampleId, int LIdIn, int LIdOut, int outStructureId1, int outStructureId2) {
  int *trustedM = (int*)malloc(sizeof(int) * M);
  if (inSize <= M) {
    opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0);
    quickSort(trustedM, 0, inSize - 1);
    opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1);
    return outStructureId1;
  }
  int realNum = SampleLoose(inStructureId, sampleId);
  int sectionNum = MultiLevelPartition(inStructureId, sampleId, LIdIn, LIdOut, realNum, P, outStructureId1, outStructureId2);
  int j = 0;
  int k, Msize;
  int hatN = (int)ceil((1 + 2 * BETA) * N);
  int sectionSize = (int)ceil(hatN / sectionNum);
  int r = (int)ceil(log(hatN / M) / log(P));
  int multiOutId = (r % 2 == 1) ? outStructureId2 : outStructureId1;
  int resOutId = (r % 2 == 0) ? outStructureId2 : outStructureId1;
  
  for (int i = 0; i < sectionNum; ++i) {
    Msize = std::min(sectionSize, hatN - i * sectionSize);
    opOneLinearScanBlock(i * sectionSize, trustedM, Msize, multiOutId, 0);
    for (k = 0; k < Msize; ++k) {
      if (trustedM[k] == DUMMY) {
        break;
      }
    }
    quickSort(trustedM, 0, k - 1);
    opOneLinearScanBlock(j, trustedM, k, resOutId, 1);
    j += k;
  }
  return resOutId;
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
void callSort(int sortId, int structureId, int paddedSize, int *resId) {
  if (sortId == 0) {
    *resId = ObliviousTightSort(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 3, structureId + 4, structureId + 5);
  } else if (sortId == 1) {
    *resId = ObliviousLooseSort(structureId, paddedSize, structureId + 1, structureId + 2, structureId + 3, structureId + 4, structureId + 5);
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


void padWithDummy(int structureId, int start, int realNum) {
  int len = BUCKET_SIZE - realNum;
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
/*
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
}*/

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
    padWithDummy(outputStructureId, bucketAddr[outputId[j]], numRow2[outputId[j]]);
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
  Bucket_x *arr = (Bucket_x*)malloc(BUCKET_SIZE * sizeof(Bucket_x));
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
    padWithDummy(structureId, bucketAddr[i], numRow1[i]);
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
}

void print(int* array, int size) {
  int i;
  for (i = 0; i < size; i++) {
    printf("%d ", array[i]);
    if (i % 5 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

void print(int **arrayAddr, int structureId, int size) {
  int i;
  if(structureSize[structureId] == 4) {
    int *addr = (int*)arrayAddr[structureId];
    for (i = 0; i < size; i++) {
      printf("%d ", addr[i]);
      if (i % 10 == 0) {
        printf("\n");
      }
    }
  } else if (structureSize[structureId] == 8) {
    Bucket_x *addr = (Bucket_x*)arrayAddr[structureId];
    for (i = 0; i < size; i++) {
      printf("(%d, %d) ", addr[i].x, addr[i].key);
      if (i % 5 == 0) {
        printf("\n");
      }
    }
  }
  printf("\n");
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



