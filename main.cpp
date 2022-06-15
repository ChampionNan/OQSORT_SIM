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

// #include "definitions.h"

#define N 2000000//1100
#define M 128 // M != B or OQSORT comes error
#define B 10

#define alpha 0.1
#define beta 0.1
#define gamma 0.1

#define NUM_STRUCTURES 10
#define MEM_IN_ENCLAVE 5
#define BLOCK_DATA_SIZE 128
#define PADDING -1
#define BUCKET_SIZE 6000//256
#define DUMMY 0xffffffff
// #define DUMMY 0x00000000
#define NULLCHAR '\0'

// structure Bucket
typedef struct {
  int x;
  int key;
} Bucket_x;


// Function
void init(void);
void print(void);
void print(int structureId);
void test(void);
void test(int structureId);
int greatestPowerOfTwoLessThan(int n);
int smallestPowerOfTwoLargerThan(int n);
void OcallReadBlock(int index, int* buffer, size_t blockSize, int structureId);
void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId);
void smallBitonicMerge(int *a, int start, int size, int flipped);
void smallBitonicSort(int *a, int start, int size, int flipped);
void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write);
void bitonicMerge(int structureId, int start, int size, int flipped, int* row1, int* row2);
void bitonicSort(int structureId, int start, int size, int flipped, int* row1, int* row2);
int callSort(int sortId, int structureId);

void padWithDummy(int structureId, int start, int realNum);
bool isTargetBitOne(int randomKey, int iter);
void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, std::vector<int> &numRows2, int outputId0, int outputId1, int iter, std::vector<int> bucketAddr, int outputStructureId);
void mergeSplit(int inputStructureId, int outputStructureId, int inputId0, int inputId1, int outputId0, int outputId1, std::vector<int> bucketAddr1, std::vector<int> bucketAddr2, std::vector<int> &numRows1, std::vector<int> &numRows2, int iter);
void kWayMergeSort(int inputStructureId, int outputStructureId, std::vector<int> &numRows1, std::vector<int> &numRows2, std::vector<int> bucketAddr);
void swapRow(Bucket_x *a, Bucket_x *b);
void swapRow(int *a, int *b);
bool cmpHelper(Bucket_x *a, Bucket_x *b);
int partition(Bucket_x *arr, int low, int high);
void quickSort(Bucket_x *arr, int low, int high);
void quickSort(int *arr, int low, int high);
void bucketSort(int inputStructureId, int bucketId, int size, int dataStart);
int bucketOSort(int structureId, int size);
int* SampleWithOutReplace(int structureId);
int PivotsSelection(int inStructureId, int samplesId, int pivotsId);
int DataPartition(int inStructureId, int outStructureId, double *quanP, int P, int levelSize);
void ObliviousSort(int inStructureId, int outStructureId);
int moveDummy(int *a, int size);


// Input & Output Array
//structureId=0, input bitonic
int* a;
//structureId=1, bucket1 in bucket sort; input
Bucket_x *bucketx1;
//structureId=2, bucket 2 in bucket sort
Bucket_x *bucketx2;
//structureId=3, write back array
int* Y;
int* arrayAddr[NUM_STRUCTURES];

int structureSize[NUM_STRUCTURES] = {sizeof(int), sizeof(Bucket_x), sizeof(Bucket_x)};
struct timeval startwtime, endwtime;
double seq_time;
int paddedSize;
const int ASCENDING  = 1;
const int DESCENDING = 0;

// structure Heap
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
  Heap(HeapNode *a, int size, int bsize) {
    heapSize = size;
    harr = a;
    int i = (heapSize - 1) / 2;
    batchSize = bsize;
    while (i >= 0) {
      Heapify(i);
      i --;
    }
  }
  
  void Heapify(int i) {
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
  
  int left(int i) {
    return (2 * i + 1);
  }
  
  int right (int i) {
    return (2 * i + 2);
  }
  
  void swapHeapNode(HeapNode *a, HeapNode *b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
  }
  
  HeapNode *getRoot() {
    return &harr[0];
  }
  
  int getHeapSize() {
    return heapSize;
  }
  
  bool reduceSizeByOne() {
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
  
  void replaceRoot(HeapNode x) {
    harr[0] = x;
    Heapify(0);
  }
  
};

// Oblivious Bucket Sort

#define DBGprint(...) { \
  fprintf(stderr, "%s: Line %d:\t", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n"); \
}


/** the main program **/
int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s n\n  where n is problem size (power of two)\n", argv[0]);
    exit(1);
  }

  //N = atoi(argv[1]);
  int addi = 0;
  if (N % BLOCK_DATA_SIZE != 0) {
    addi = ((N / BLOCK_DATA_SIZE) + 1) * BLOCK_DATA_SIZE - N;
  }
  
  
  // TODO: test bitonicSort
  // init();
  //gettimeofday (&startwtime, NULL);
  // smallBitonicSort(a, 0, paddedSize, 0);
  // callSort(3, 0);
  //gettimeofday (&endwtime, NULL);
  //seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
             // + endwtime.tv_sec - startwtime.tv_sec);
  //printf("Imperative wall clock time = %f\n", seq_time);
  // test();
  
  // TODO: test bucketOSort
  srand((unsigned)time(NULL));
  int bucketNum = smallestPowerOfTwoLargerThan(ceil(2.0 * N / BUCKET_SIZE));
  int bucketSize = bucketNum * BUCKET_SIZE;
  std::cout<<"TOTAL BUCKET SIZE: " << bucketSize<<std::endl;
  bucketx1 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
  bucketx2 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
  memset(bucketx1, '\0', bucketSize*sizeof(Bucket_x));
  memset(bucketx2, '\0', bucketSize*sizeof(Bucket_x));
  arrayAddr[1] = (int*)bucketx1;
  arrayAddr[2] = (int*)bucketx2;
  
  a = (int *) malloc(N * sizeof(int));
  arrayAddr[0] = a;
  paddedSize = N;
  init();
  /*
  std::cout << "=======InitialA=======\n";
  print(0);
  std::cout << "=======InitialA=======\n";*/
  int resId = callSort(2, 1);
  paddedSize = N;
  test(resId);
  // print();
}


int greatestPowerOfTwoLessThan(int n) {
    int k = 1;
    while (k > 0 && k < n) {
        k = k << 1;
    }
    return k >> 1;
}

int smallestPowerOfTwoLargerThan(int n) {
  int k = 1;
  while (k > 0 && k < n) {
    k = k << 1;
  }
  return k;
}

// TODO: Set this function as OCALL
void freeAllocate(int structureId, int size) {
  // 1. Free arrayAddr[structureId]
  if (arrayAddr[structureId]) {
    free(arrayAddr[structureId]);
  }
  // 2. malloc new asked size (allocated in outside)
  int *addr = (int*)malloc(size * sizeof(int));
  memset(addr, DUMMY, size * sizeof(int));
  // 3. assign malloc address to arrayAddr
  arrayAddr[structureId] = addr;
  return ;
}

void OcallReadBlock(int index, int* buffer, size_t blockSize, int structureId) {
  if (blockSize == 0) {
    //printf("Unknown data size\n");
    return;
  }
  // memcpy(buffer, &(arrayAddr[structureId][index]), blockSize * structureSize[structureId]);
  memcpy(buffer, arrayAddr[structureId] + index, blockSize * structureSize[structureId]);
}

// index: 数据类型为int时的offset，字节数
void OcallWriteBlock(int index, int* buffer, size_t blockSize, int structureId) {
  if (blockSize == 0) {
    //printf("Unknown data size\n");
    return;
  }/*
  if (typeid(structureId) == typeid(int*)) {
    memcpy(arrayAddr[structureId] + index, buffer, blockSize * structureSize[structureId]);
  } else if (typeid(structureId) == typeid(Bucket_x*)) {
    Bucket_x *addr = (Bucket_x*)arrayAddr[structureId];
    memcpy(addr + index, buffer, blockSize * structureSize[structureId]);
  }*/
  memcpy(arrayAddr[structureId] + index, buffer, blockSize * structureSize[structureId]);
  // memcpy(&(arrayAddr[structureId][index]), buffer, blockSize * structureSizfine[structureId]);
}

void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write) {
  if (!write) {
    OcallReadBlock(index, block, blockSize, structureId);
  } else {
    OcallWriteBlock(index, block, blockSize, structureId);
  }
  return;
}

int SampleWithOutReplace(int inStructureId, int samplesId) {
  int n_prime = (int)ceil(alpha * N);
  int alphaM2 = (int)ceil(2 * alpha * M);
  int boundary = (int)ceil(N/M);
  int Msize;
  int realNum = 0; // #pivots
  int writeBacksize = 0; // #y's write back size
  int writeBackstart = 0;
  int readStart = 0;
  int *y = (int*)malloc(M * sizeof(int));
  int *trustedMemory = (int*)malloc(M * sizeof(int));
  
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(M, N - i * M);
    opOneLinearScanBlock(readStart, trustedMemory, Msize, inStructureId, 0);
    readStart += Msize;
    for (int j = 0; j < Msize; ++j) {
      std::random_device rd;
      std::mt19937_64 generator(rd());
      double rate = alpha;
      std::bernoulli_distribution b(rate);
      if (b(generator)) {
        n_prime --;
      } else {
        trustedMemory[j] = DUMMY;
      }
    }
    realNum += moveDummy(trustedMemory, Msize);
    
    if (writeBacksize + alphaM2 >= M) {
      opOneLinearScanBlock(writeBackstart, y, writeBacksize, samplesId, 1);
      // only update here
      writeBackstart += writeBacksize;
      writeBacksize = 0;
      memcpy(y, trustedMemory, alphaM2);
    } else {
      memcpy(y + writeBackstart, trustedMemory, alphaM2);
      writeBacksize += alphaM2;
    }
  }
  
  free(trustedMemory);
  free(y);
  if (realNum < (int)(alpha * N)) {
    return -1;
  }
  return realNum;
}

// TODO: at the end, free y & pivots
// TODO: M has the same size with BLOCK_DATA_SIZE
// TODO: dummy elements in the sampled should be placed at the end
int PivotsSelection(int inStructureId, int samplesId, int pivotsId) {
  int res = -1;
  // If fail, repeat
  while (res == -1) {
    res = SampleWithOutReplace(inStructureId, samplesId);
  }
  // sort pivots using bitonic sort, we can use sort algorithm randomly here
  int *row1 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  int *row2 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  bitonicSort(samplesId, 0, res, 0, row1, row1);
  free(row1);
  free(row2);
  // int pSize = (int)ceil(N / M);
  int *p = (int*)malloc(M * sizeof(int));
  double j = alpha * M;
  int k = 0;
  // TODO: why still return *p if the pivots are stored in pivotsId
  int totalK = 0; // #pivots
  int end = (int)ceil(alpha * N / M);
  int *trustedMemory = (int*)malloc(M * sizeof(int));
  int writeBackstart = 0;
  int readStart = 0;
  
  for (int i = 0; i < end; ++i) {
    int Msize = std::min(M, (int)ceil(alpha * N) - i * M);
    opOneLinearScanBlock(readStart, trustedMemory, Msize, samplesId, 0);
    readStart += Msize;
    while (j < M) {
      int indexj = (int)floor(j);
      if (k >= M) {
        opOneLinearScanBlock(writeBackstart, p, M, pivotsId, 1);
        writeBackstart += M;
        totalK += M;
        k -= M;
      }
      p[k++] = trustedMemory[indexj];
      j += alpha * M;
    }
    j -= M;
  }
  // TODO: Left write back
  opOneLinearScanBlock(writeBackstart, p, k, pivotsId, 1);
  totalK += k;
  free(trustedMemory);
  return totalK; // p is not the full pivots
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


// TODO: change memory to P with circular division (Finished)
int DataPartition(int inStructureId, int outStructureId, double *quanP, int P, int levelSize) {
  int M_prime = (int)ceil(M / (1 + 2 * beta));
  int XiSize = (int)ceil(N * B / M_prime);
  // 1. initial Xi
  int mallocSize = P * XiSize;
  freeAllocate(outStructureId, mallocSize);
  // 2. for j
  int blockSize;
  int end = (int)ceil(N / M_prime);
  int psecSize = (int)ceil(M / P);
  int readStart = 0;
  // Inidicate #elem in bucket(#bucket = P)
  int *offset = (int*)malloc(P * sizeof(int));
  memset(offset, 0, P * sizeof(int));
  int *writeBackOffset = (int*)malloc(P * sizeof(int));
  memset(writeBackOffset, 0, P * sizeof(int));
  int *trustedMemory = (int*)malloc(M_prime * sizeof(int));
  int *xiSec = (int*)malloc(psecSize * sizeof(int));
  memset(xiSec, DUMMY, psecSize * sizeof(int));
  int retFlag = 1;
  // TODO: check fail condition: move more than M/P elements
  for (int j = 0; j < end; ++j) {
    int jstart = j * psecSize;
    blockSize = std::min(M_prime, levelSize - j * M_prime);
    opOneLinearScanBlock(readStart, trustedMemory, blockSize, inStructureId, 0);
    readStart += blockSize;
    for (int i = 0; i < P; ++i) {
      int istart = i * XiSize;
      for (int k = 0; k < blockSize; ++k) {
        int x = trustedMemory[k];
        if (i == 0 && x <= quanP[i]) {
          if (offset[i] > psecSize) {
            DBGprint("DataPartition Fail");
            retFlag = 0;
          }
          xiSec[istart + jstart + offset[i]] = x;
          offset[i] ++;
        } else if ((i != (P - 1)) && (x > quanP[i]) && (x <= quanP[i + 1])){
          if (offset[i + 1] > psecSize) {
            DBGprint("DataPartition Fail");
            retFlag = 0;
          }
          xiSec[istart + jstart + offset[i + 1]] = x;
          offset[i + 1] ++;
        } else if (i == (P - 1) && x > quanP[i]) {
          if (offset[0] > psecSize) {
            DBGprint("DataPartition Fail");
            retFlag = 0;
          }
          xiSec[istart + jstart + offset[0]] = x;
          offset[0] ++;
        } else {
          DBGprint("partion section error");
          retFlag = 0;
        }
      } // end-k
      // TODO: which size write back?
      opOneLinearScanBlock(istart + writeBackOffset[i], xiSec, offset[i], outStructureId, 1);
      writeBackOffset[i] += offset[i];
      offset[i] = 0;
      memset(xiSec, DUMMY, psecSize * sizeof(int));
    } // end-i
    
  } // end-j
  // TODO: Free data structure
  free(offset);
  free(writeBackOffset);
  free(trustedMemory);
  free(xiSec);
  return retFlag;
}

double quantileCal(int *a, int size, double rate) {
  assert(rate >= 0.0 && rate <= 1.0);
  double id = (size - 1) * rate;
  int lo = floor(id);
  int hi = ceil(id);
  int qs = a[lo];
  double h = id - lo;
  return (1.0 - h) * qs + h * a[hi];
}

// structureId=-1 -> use y in pivots selection use other sorting agorithm
// TODO: 需要使用MEM_IN_ENCLAVE来做enclave memory的约束? how
// TODO: 主要是用来限制oponelinear一次读取进来的数据大小
int ObliviousSort(int inStructureId, int inSize, int sampleId, int pivotsId, int outStructureId) {
  int *trustedMemory = NULL;
  if (N <= M) {
    trustedMemory = (int*)malloc(N);
    // TODO: use sort algorithm
    // TODO: Change blockSize to BLOCK_DATA_SIZE
    opOneLinearScanBlock(0, trustedMemory, N, inStructureId, 0);
    quickSort(trustedMemory, 0, N - 1);
    opOneLinearScanBlock(0, trustedMemory, N, outStructureId, 1);
    free(trustedMemory);
    return 1;
  }
  // TODO: p is the address of pivots, which need to free
  int numPivots = -1;
  numPivots = PivotsSelection(inStructureId, sampleId, pivotsId);
  // Fisher-Yates shuffle
  // TODO: 这里是使用paddwithdummy还是shuffle一部分数据？
  trustedMemory = (int*)malloc(2 * B * sizeof(int));
  int iEnd = (int)ceil(N/B) - 2;
  for (int i = 0; i <= iEnd; ++i) {
    std::default_random_engine generator;
    int right = (int)ceil(N/B);
    std::uniform_int_distribution<int> distribution(i, right - 1);
    int j = distribution(generator);
    int jSize = B;
    if (j == right - 1) {
      jSize = N - (right - 1) * B;
    }
    opOneLinearScanBlock(i * B, trustedMemory, jSize, inStructureId, 0);
    opOneLinearScanBlock(j * B, &trustedMemory[B], jSize, inStructureId, 0);
    opOneLinearScanBlock(i * B, &trustedMemory[B], jSize, inStructureId, 1);
    opOneLinearScanBlock(j * B, trustedMemory, jSize, inStructureId, 1);
  }
  // double M_prime = ceil(M / (1 + 2 * beta));
  int r = (int)ceil(log(N / M) / log(numPivots / (1 + 2 * beta)));
  int levelSize = 0;
  // TODO: 这里使用inStructId和outStructId两个数组进行轮换，以实现r层数据的partition
  // TODO: should pad with dummy
  for (int i = 0; i < r; ++i) {
    int jEnd = (int)ceil(pow(M/B, i));
    int W = (int)((N/M)/jEnd);
    int *p = (int*)malloc(W * sizeof(int));
    double *quanP = (double*)malloc(sizeof(double) * jEnd);
    for (int j = 0; j < jEnd; ++j) {
      int wSize = std::min(W, numPivots - j * W);
      opOneLinearScanBlock(j * W, p, wSize, pivotsId, 0);
      // TODO: use i 的奇偶性来进行区分
      quanP[j] = quantileCal(&p[j * W], wSize, M / B);
    }
    
    int XiSize = (int)ceil(N * B * (1 + 2 * beta) / M);
    levelSize = jEnd * XiSize; // use two part both ceiling
    int flag = 0;
    if (i % 2) {
      while (!flag) {
        flag = DataPartition(inStructureId, outStructureId, quanP, jEnd, levelSize);
      }
    } else {
      while (!flag) {
        flag = DataPartition(outStructureId, inStructureId, quanP, jEnd, levelSize);
      }
    }
  }
  int jEnd = (int)pow(M/B, r);
  int blockSize = (int)ceil(M/B);
  int totalReal = 0; // use for write back address
  // TODO: previous pointed should be free
  trustedMemory = (int*)malloc((int)ceil(M/B) * sizeof(int));
  for (int j = 0; j < jEnd; ++j) {
    int readSize = std::min(blockSize, levelSize - j * blockSize);
    if (r % 2) {
      opOneLinearScanBlock(j * blockSize, trustedMemory, readSize, outStructureId, 0);
      int real = moveDummy(trustedMemory, readSize);
      quickSort(trustedMemory, 0, real - 1);
      opOneLinearScanBlock(totalReal, trustedMemory, real, inStructureId, 1);
      totalReal += real;
    } else {
      opOneLinearScanBlock(j * blockSize, trustedMemory, readSize, inStructureId, 0);
      int real = moveDummy(trustedMemory, readSize);
      quickSort(trustedMemory, 0, real - 1);
      opOneLinearScanBlock(totalReal, trustedMemory, readSize, outStructureId, 1);
      totalReal += real;
    }
  }
  assert(totalReal == N && "Output array number error");
  return r % 2 == 0; // return 1->outId; 0->inId
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
  } else if (size < MEM_IN_ENCLAVE) {
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
  } else if (size < MEM_IN_ENCLAVE) {
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
int callSort(int sortId, int *structureId) {
  // bitonic sort
  int size = paddedSize / BLOCK_DATA_SIZE;
  printf("size: %d %d\n", paddedSize, size);
  if (sortId == 1) {
    int inStructureId = structureId[0];
    int sampleId = structureId[1];
    int pivotsId = structureId[2];
    int outStructureId = structureId[3];
    return ObliviousSort(inStructureId, N, sampleId, pivotsId, outStructureId);
  }
  if (sortId == 2) {
    return bucketOSort(structureId[0], N);
  }
  if (sortId == 3) {
    int *row1 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
    int *row2 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
    bitonicSort(structureId[0], 0, size, 0, row1, row2);
    free(row1);
    free(row2);
    return -1;
  }
  return -1;
}


void padWithDummy(int structureId, int start, int realNum) {
  //int blockSize = structureSize[structureId];
  int len = BUCKET_SIZE - realNum;
  if (len <= 0) {
    return ;
  }
  int *junk = (int*)malloc(len * sizeof(Bucket_x));
  // memset(junk, 0xff, blockSize * len);
  for (int i = 0; i < len * 2; ++i) {
    junk[i] = -1;
  }
  
  opOneLinearScanBlock(2 * (start + realNum), (int*)junk, len, structureId, 1);
  free(junk);
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

bool isTargetBitOne(int randomKey, int iter) {
  assert(iter >= 1);
  return (randomKey & (0x01 << (iter - 1))) == 0 ? false : true;
}

void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, std::vector<int> &numRows2, int outputId0, int outputId1, int iter, std::vector<int> bucketAddr, int outputStructureId) {
  // write back standard
  int batchSize = 1; // 8192
  Bucket_x *buf0 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
  Bucket_x *buf1 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
  int counter0 = 0, counter1 = 0;
  int randomKey = 0;
  
  for (int i = 0; i < inputBufferLen; ++i) {
    if ((inputBuffer[i].key != DUMMY) && (inputBuffer[i].x != DUMMY)) {
      randomKey = inputBuffer[i].key;
      
      if (isTargetBitOne(randomKey, iter + 1)) {
        buf1[counter1 % batchSize] = inputBuffer[i];
        counter1 ++;
        if (counter1 % batchSize == 0) {
          // int start = (counter1 / batchSize - 1) * batchSize;
          opOneLinearScanBlock(2 * (bucketAddr[outputId1] +  numRows2[outputId1]), (int*)buf1, (size_t)batchSize, outputStructureId, 1);
          numRows2[outputId1] += batchSize;
          memset(buf1, NULLCHAR, batchSize * sizeof(Bucket_x));
        }
      } else {
        buf0[counter0 % batchSize] = inputBuffer[i];
        counter0 ++;
        if (counter0 % batchSize == 0) {
          // int start = (counter0 / batchSize - 1) * batchSize;
          opOneLinearScanBlock(2 * (bucketAddr[outputId0] + numRows2[outputId0]), (int*)buf0, (size_t)batchSize, outputStructureId, 1);
          numRows2[outputId0] += batchSize;
          memset(buf0, NULLCHAR, batchSize * sizeof(Bucket_x));
        }
      }
    }
  }
  
  // int start = (counter1 / batchSize - 1) * batchSize;
  opOneLinearScanBlock(2 * (bucketAddr[outputId1] + numRows2[outputId1]), (int*)buf1, (size_t)(counter1 % batchSize), outputStructureId, 1);
  numRows2[outputId1] += counter1 % batchSize;
  // start = (counter0 / batchSize - 1) * batchSize;
  opOneLinearScanBlock(2 * (bucketAddr[outputId0] + numRows2[outputId0]), (int*)buf0, (size_t)(counter0 % batchSize), outputStructureId, 1);
  numRows2[outputId0] += counter0 % batchSize;
  
  free(buf0);
  free(buf1);
}

// inputId: start index for inputStructureId
// numRow1: input bucket length, numRow2: output bucket length
void mergeSplit(int inputStructureId, int outputStructureId, int inputId0, int inputId1, int outputId0, int outputId1, std::vector<int> bucketAddr1, std::vector<int> bucketAddr2, std::vector<int> &numRows1, std::vector<int> &numRows2, int iter) {
  Bucket_x *inputBuffer = (Bucket_x*)malloc(sizeof(Bucket_x) * BUCKET_SIZE);
  // BLOCK#0
  opOneLinearScanBlock(2 * bucketAddr1[inputId0], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
  mergeSplitHelper(inputBuffer, numRows1[inputId0], numRows2, outputId0, outputId1, iter, bucketAddr2, outputStructureId);
  if (numRows2[outputId0] > BUCKET_SIZE || numRows2[outputId1] > BUCKET_SIZE) {
    DBGprint("overflow error during merge split!\n");
  }
  
  // BLOCK#1
  opOneLinearScanBlock(2 * bucketAddr1[inputId1], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
  mergeSplitHelper(inputBuffer, numRows1[inputId1], numRows2, outputId0, outputId1, iter, bucketAddr2, outputStructureId);
  
  if (numRows2[outputId0] > BUCKET_SIZE || numRows2[outputId1] > BUCKET_SIZE) {
    DBGprint("overflow error during merge split!\n");
  }
  
  padWithDummy(outputStructureId, bucketAddr2[outputId1], numRows2[outputId1]);
  padWithDummy(outputStructureId, bucketAddr2[outputId0], numRows2[outputId0]);
  
  free(inputBuffer);
}

void kWayMergeSort(int inputStructureId, int outputStructureId, std::vector<int> &numRows1, std::vector<int> &numRows2, std::vector<int> bucketAddr) {
  int mergeSortBatchSize = 256; // 256
  int writeBufferSize = 8192; // 8192
  int numWays = (int)numRows1.size();
  HeapNode inputHeapNodeArr[numWays];
  int totalCounter = 0;
  std::vector<int> readBucketAddr(bucketAddr);
  int writeBucketAddr = 0;
  
  for (int i = 0; i < numWays; ++i) {
    HeapNode node;
    node.data = (Bucket_x*)malloc(mergeSortBatchSize * sizeof(Bucket_x));
    node.bucketIdx = i;
    node.elemIdx = 0;
    opOneLinearScanBlock(2 * readBucketAddr[i], (int*)node.data, (size_t)std::min(mergeSortBatchSize, numRows1[i]), inputStructureId, 0);
    inputHeapNodeArr[i] = node;
    // Update data count
    readBucketAddr[i] += std::min(mergeSortBatchSize, numRows1[i]);
  }
  
  Heap heap(inputHeapNodeArr, numWays, mergeSortBatchSize);
  // DBGprint("init heap success");
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
      numRows2[temp->bucketIdx] += writeBufferSize;
      writeBufferCounter = 0;
    }
    
    // re-get bucketIdx mergeSortBatchSize data, juct compare certain index data
    if (temp->elemIdx < numRows1[temp->bucketIdx] && (temp->elemIdx % mergeSortBatchSize) == 0) {
      
      opOneLinearScanBlock(2 * readBucketAddr[temp->bucketIdx], (int*)(temp->data), (size_t)std::min(mergeSortBatchSize, numRows1[temp->bucketIdx]-temp->elemIdx), inputStructureId, 0);
      
      readBucketAddr[temp->bucketIdx] += std::min(mergeSortBatchSize, numRows1[temp->bucketIdx]-temp->elemIdx);
      heap.Heapify(0);
      // one bucket data is empty
    } else if (temp->elemIdx >= numRows1[temp->bucketIdx]) {
      bool res = heap.reduceSizeByOne();
      if (!res) {
        break;
      }
    } else {
      heap.Heapify(0);
    }
  }
  opOneLinearScanBlock(2 * writeBucketAddr, (int*)writeBuffer, (size_t)writeBufferCounter, outputStructureId, 1);
  numRows2[-1] += writeBufferCounter;
  free(writeBuffer);
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

bool cmpHelper(Bucket_x *a, Bucket_x *b) {
  return (a->x > b->x) ? true : false;
}

bool cmpHelper(int *a, int *b) {
  return (*a > *b) ? true : false;
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

void bucketSort(int inputStructureId, int bucketId, int size, int dataStart) {
  Bucket_x *arr = (Bucket_x*)malloc(BUCKET_SIZE * sizeof(Bucket_x));
  opOneLinearScanBlock(2 * dataStart, (int*)arr, (size_t)size, inputStructureId, 0);
  quickSort(arr, 0, size - 1);
  opOneLinearScanBlock(2 * dataStart, (int*)arr, (size_t)size, inputStructureId, 1);
  free(arr);
}

// size = #inputs real size
int bucketOSort(int structureId, int size) {
  int bucketNum = smallestPowerOfTwoLargerThan(ceil(2.0 * size / BUCKET_SIZE));
  paddedSize = bucketNum * BUCKET_SIZE;
  std::vector<int> bucketAddr1(bucketNum, -1);
  std::vector<int> bucketAddr2(bucketNum, -1);
  std::vector<int> numRows1(bucketNum, 0);
  std::vector<int> numRows2(bucketNum, 0);
  
  int ranBinAssignIters = log2(bucketNum) - 1;
  // int elementsPerBucket = (int)floor(size / bucketNum);
  // int reminder = size - elementsPerBucket * bucketNum;
  // std::vector<int> bucketElem(bucketNum);
  for (int i = 0; i < bucketNum; ++i) {
    // bucketElem[i] = elementsPerBucket + (i < reminder ? 1 : 0);
    // bucketAddr: #elem in each bucket
    bucketAddr1[i] = i * BUCKET_SIZE;
    bucketAddr2[i] = i * BUCKET_SIZE;
  }
  
  // TODO: Assign each element in X a uniformly random key (Finished)
  srand((unsigned)time(NULL));
  Bucket_x *trustedMemory = (Bucket_x*)malloc(BLOCK_DATA_SIZE * sizeof(Bucket_x));
  int *inputTrustMemory = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  
  int total = 0;
  
  // read input & generate randomKey, write to designed bucket
  for (int i = 0; i < size; i += BLOCK_DATA_SIZE) {
    // TODO: 973 later x become all 0
    opOneLinearScanBlock(i, inputTrustMemory, std::min(BLOCK_DATA_SIZE, size - i), structureId - 1, 0);
    int randomKey;
    for (int j = 0; j < std::min(BLOCK_DATA_SIZE, size - i); ++j) {
      randomKey = rand();
      trustedMemory[j].x = inputTrustMemory[j];
      trustedMemory[j].key = randomKey;
      // TODO: improve
      int offset = bucketAddr1[(i + j) % bucketNum] + numRows1[(i + j) % bucketNum];
      opOneLinearScanBlock(offset * 2, (int*)(&trustedMemory[j]), (size_t)1, structureId, 1);/*
      std::cout << "=======RanDom=======\n";
      paddedSize = bucketNum * BUCKET_SIZE;
      std::cout<<"Offset: "<<offset<<std::endl;
      std::cout<<trustedMemory[j].x<<" "<<trustedMemory[j].key<<std::endl;
      print(structureId);
      std::cout << "=======Random=======\n";*/
      paddedSize = bucketNum * BUCKET_SIZE;
      numRows1[(i + j) % bucketNum] ++;
    }
    total += std::min(BLOCK_DATA_SIZE, size - i);
  }
  free(trustedMemory);
  free(inputTrustMemory);
  
  for (int i = 0; i < bucketNum; ++i) {
    DBGprint("currently bucket %d has %d records/%d", i, numRows1[i], BUCKET_SIZE);
    padWithDummy(structureId, bucketAddr1[i], numRows1[i]);
    
  }
  /*
  std::cout << "=======Initial=======\n";
  paddedSize = bucketNum * BUCKET_SIZE;
  print(structureId);
  std::cout << "=======Initial=======\n";*/
  
  for (int i = 0; i < ranBinAssignIters; ++i) {
    // data in Array1, update numRows2
    if (i % 2 == 0) {
      for (int j = 0; j < bucketNum / 2; ++j) {
        int jj = (j / (int)pow(2, i)) * (int)pow(2, i);
        mergeSplit(structureId, structureId + 1, j + jj, j + jj + (int)pow(2, i), 2 * j, 2 * j + 1, bucketAddr1, bucketAddr2, numRows1, numRows2, i);
      }
      int count = 0;
      for (int k = 0; k < bucketNum; ++k) {
        numRows1[k] = 0;
        count += numRows2[k];
      }
      DBGprint("after %dth merge split, we have %d tuples\n", i, count);
    } else {
      for (int j = 0; j < bucketNum / 2; ++j) {
        int jj = (j / (int)pow(2, i)) * (int)pow(2, i);
        // TODO: scan
        mergeSplit(structureId + 1, structureId, j + jj, j + jj + (int)pow(2, i), 2 * j, 2 * j + 1, bucketAddr2, bucketAddr1, numRows2, numRows1, i);
      }
      int count = 0;
      for (int k = 0; k < bucketNum; ++k) {
        numRows2[k] = 0;
        count += numRows1[k];
      }
      DBGprint("after %dth merge split, we have %d tuples\n", i, count);
    }
    DBGprint("\n\n Finish random bin assignment iter%dth out of %d\n\n", i, ranBinAssignIters);
  }
  
  int resultId = 0;
  if (ranBinAssignIters % 2 == 0) {
    for (int i = 0; i < bucketNum; ++i) {
      bucketSort(structureId, i, numRows1[i], bucketAddr1[i]);
    }
    kWayMergeSort(structureId, structureId + 1, numRows1, numRows2, bucketAddr1);
    resultId = structureId + 1;
  } else {
    for (int i = 0; i < bucketNum; ++i) {
      bucketSort(structureId + 1, i, numRows2[i], bucketAddr2[i]);
    }
    kWayMergeSort(structureId + 1, structureId, numRows2, numRows1, bucketAddr2);
    resultId = structureId;
  }
  DBGprint("SOrtId: %d", resultId);
  return resultId;
}



/** -------------- SUB-PROCEDURES  ----------------- **/

/** procedure test() : verify sort results **/
void test() {
  int pass = 1;
  int i;
  for (i = 1; i < paddedSize; i++) {
    pass &= (a[i-1] <= a[i]);
  }

  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
    // print();
}

void test(int structureId) {
  int pass = 1;
  int i;
  // print(structureId);
  for (i = 1; i < paddedSize; i++) {
    pass &= (((Bucket_x*)arrayAddr[structureId])[i-1].x <= ((Bucket_x*)arrayAddr[structureId])[i].x);
  }
  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
}


/** procedure init() : initialize array "a" with data **/
void init() {
  int i;
  for (i = 0; i < paddedSize; i++) {
    // a[i] = rand() % N; // (N - i);
    a[i] = (paddedSize - i);
  }
}

/** procedure  print() : print array elements **/
void print() {
  int i;
  for (i = 0; i < paddedSize; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");
}

// Judge by structure size
void print(int structureId) {
  int i;
  for (i = 0; i < paddedSize; i++) {
    if(structureSize[structureId] == 4) {
      // int
      int *addr = (int*)arrayAddr[structureId];
      printf("%d ", addr[i]);
    } else if (structureSize[structureId] == 8) {
      //Bucket
      Bucket_x *addr = (Bucket_x*)arrayAddr[structureId];
      printf("(%d, %d), ", addr[i].x, addr[i].key);
    }
  }
  printf("\n");
}



