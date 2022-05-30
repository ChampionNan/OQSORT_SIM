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

// #include "definitions.h"

#define N 20//1100
#define M 128
#define B 10

#define alpha 0.1
#define beta 0.1
#define gamma 0.1

#define NUM_STRUCTURES 10
#define MEM_IN_ENCLAVE 5
#define BLOCK_DATA_SIZE 128
#define PADDING -1
#define BUCKET_SIZE 6//256
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
void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, std::vector<int> numRows2, int outputId0, int outputId1, int iter, std::vector<int> bucketAddr, int outputStructureId);
void mergeSplit(int inputStructureId, int outputStructureId, int inputId0, int inputId1, int outputId0, int outputId1, std::vector<int> bucketAddr1, std::vector<int> bucketAddr2, std::vector<int> numRows1, std::vector<int> numRows2, int iter);
void kWayMergeSort(int inputStructureId, int outputStructureId, std::vector<int> bucketSize, std::vector<int> bucketAddr);
void swapRow(Bucket_x *a, Bucket_x *b);
bool cmpHelper(Bucket_x *a, Bucket_x *b);
int partition(Bucket_x *arr, int low, int high);
void quickSort(Bucket_x *arr, int low, int high);
void bucketSort(int inputStructureId, int bucketId, int size, int dataStart);
int bucketOSort(int structureId, int size);


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
  Heap(HeapNode *a, int size, int batchSize) {
    heapSize = size;
    harr = a;
    int i = (heapSize - 1) / 2;
    batchSize = batchSize;
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
  std::cout << "=======Test=======\n";
  print(0);
  print(1);
  int *addr = arrayAddr[1];
  int *a = (int*)malloc(sizeof(Bucket_x) * 2);
  a[0] = -1;
  a[1] = -1;
  a[2] = -1;
  a[3] = -1;
  memcpy(addr + 2, a, 2 * sizeof(Bucket_x));
  print(1);
  std::cout << "=======Test=======\n";*/
  
  std::cout << "=======InitialA=======\n";
  print(0);
  std::cout<<std::endl;
  std::cout << "=======InitialA=======\n";
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
int callSort(int sortId, int structureId) {
  // bitonic sort
  int size = paddedSize / BLOCK_DATA_SIZE;
  printf("size: %d %d\n", paddedSize, size);
  if (sortId == 2) {
    return bucketOSort(structureId, N);
  }
  if (sortId == 3) {
    int *row1 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
    int *row2 = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
    bitonicSort(structureId, 0, size, 0, row1, row2);
    return -1;
  }
  return -1;
}

// Oblivious Bucket Sort

#define DBGprint(...) { \
  fprintf(stderr, "%s: Line %d:\t", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n"); \
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
  
  opOneLinearScanBlock(1 * (start + realNum) + 1, (int*)junk, (len)/2, structureId, 1);
  free(junk);
}

void moveDummy(int structureId, int start) {
  Bucket_x *temp = (Bucket_x*)malloc(BUCKET_SIZE * sizeof(Bucket_x));
  opOneLinearScanBlock(start, (int*)temp, BUCKET_SIZE, structureId, 0);
  int k = 0;
  for (int i = 0; i < BUCKET_SIZE; ++i) {
    if (temp[i].x != DUMMY) {
      if (i != k) {
        swapRow(&temp[i], &temp[k++]);
      } else {
        k++;
      }
    }
  }
  opOneLinearScanBlock(start, (int*)temp, BUCKET_SIZE, structureId, 1);
  free(temp);
}

bool isTargetBitOne(int randomKey, int iter) {
  assert(iter >= 1);
  return (randomKey & (0x01 << (32 - iter))) == 0 ? false : true;
}

void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, std::vector<int> numRows2, int outputId0, int outputId1, int iter, std::vector<int> bucketAddr, int outputStructureId) {
  // write back standard
  int batchSize = 128; // 8192
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
          int start = (counter1 / batchSize - 1) * batchSize;
          opOneLinearScanBlock(bucketAddr[outputId1] + start, (int*)buf1, (size_t)batchSize, outputStructureId, 1);
          numRows2[outputId1] += batchSize;
          memset(buf1, NULLCHAR, batchSize * sizeof(Bucket_x));
        }
      } else {
        buf0[counter0 % batchSize] = inputBuffer[i];
        counter0 ++;
        if (counter0 % batchSize == 0) {
          int start = (counter0 / batchSize - 1) * batchSize;
          opOneLinearScanBlock(bucketAddr[outputId0] + start, (int*)buf0, (size_t)batchSize, outputStructureId, 1);
          numRows2[outputId0] += batchSize;
          memset(buf0, NULLCHAR, batchSize * sizeof(Bucket_x));
        }
      }
    }
  }
  
  int start = (counter1 / batchSize - 1) * batchSize;
  opOneLinearScanBlock(bucketAddr[outputId1] + start, (int*)buf1, (size_t)(counter1 % batchSize), outputStructureId, 1);
  numRows2[outputId1] += counter1 % batchSize;
  start = (counter0 / batchSize - 1) * batchSize;
  opOneLinearScanBlock(bucketAddr[outputId0] + start, (int*)buf0, (size_t)(counter0 % batchSize), outputStructureId, 1);
  numRows2[outputId0] += counter0 % batchSize;
  
  free(buf0);
  free(buf1);
}

// inputId: start index for inputStructureId
// numRow1: input bucket length, numRow2: output bucket length
void mergeSplit(int inputStructureId, int outputStructureId, int inputId0, int inputId1, int outputId0, int outputId1, std::vector<int> bucketAddr1, std::vector<int> bucketAddr2, std::vector<int> numRows1, std::vector<int> numRows2, int iter) {
  Bucket_x *inputBuffer = (Bucket_x*)malloc(sizeof(Bucket_x) * BUCKET_SIZE);
  // BLOCK#0
  opOneLinearScanBlock(bucketAddr1[inputId0], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
  mergeSplitHelper(inputBuffer, numRows1[inputId0], numRows2, outputId0, outputId1, iter, bucketAddr2, outputStructureId);
  if (numRows2[outputId0] > BUCKET_SIZE || numRows2[outputId1] > BUCKET_SIZE) {
    DBGprint("overflow error during merge split!\n");
  }
  
  // BLOCK#1
  opOneLinearScanBlock(bucketAddr1[inputId1], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
  mergeSplitHelper(inputBuffer, numRows1[inputId1], numRows2, outputId0, outputId1, iter, bucketAddr2, outputStructureId);
  if (numRows2[outputId0] > BUCKET_SIZE || numRows2[outputId1] > BUCKET_SIZE) {
    DBGprint("overflow error during merge split!\n");
  }
  padWithDummy(outputStructureId, bucketAddr2[outputId1], numRows2[outputId1]);
  padWithDummy(outputStructureId, bucketAddr2[outputId0], numRows2[outputId0]);
  
  free(inputBuffer);
}

void kWayMergeSort(int inputStructureId, int outputStructureId, std::vector<int> numRows1, std::vector<int> numRows2, std::vector<int> bucketAddr) {
  int mergeSortBatchSize = 64; // 256
  int writeBufferSize = 128; // 8192
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
    opOneLinearScanBlock(readBucketAddr[i], (int*)node.data, (size_t)std::min(mergeSortBatchSize, numRows1[i]), inputStructureId, 0);
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
      opOneLinearScanBlock(writeBucketAddr, (int*)temp, (size_t)writeBufferSize, outputStructureId, 1);
      writeBucketAddr += writeBufferSize;
      numRows2[temp->bucketIdx] += writeBufferSize;
      writeBufferCounter = 0;
    }
    
    // re-get bucketIdx mergeSortBatchSize data, juct compare certain index data
    if (temp->elemIdx < numRows1[temp->bucketIdx] && (temp->elemIdx % mergeSortBatchSize) == 0) {
      opOneLinearScanBlock(readBucketAddr[temp->bucketIdx], (int*)temp->data, (size_t)std::min(mergeSortBatchSize, numRows1[temp->bucketIdx]-temp->elemIdx), inputStructureId, 0);
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
  opOneLinearScanBlock(writeBucketAddr, (int*)writeBuffer, (size_t)writeBufferCounter, outputStructureId, 1);
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

bool cmpHelper(Bucket_x *a, Bucket_x *b) {
  return (a->x > b->x) ? true : false;
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

void quickSort(Bucket_x *arr, int low, int high) {
  if (high > low) {
    int mid = partition(arr, low, high);
    quickSort(arr, low, mid - 1);
    quickSort(arr, mid + 1, high);
  }
}

void bucketSort(int inputStructureId, int bucketId, int size, int dataStart) {
  Bucket_x *arr = (Bucket_x*)malloc(BUCKET_SIZE * sizeof(Bucket_x));
  opOneLinearScanBlock(dataStart, (int*)arr, (size_t)size, inputStructureId, 0);
  quickSort(arr, 0, size - 1);
  opOneLinearScanBlock(dataStart, (int*)arr, (size_t)size, inputStructureId, 1);
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
      randomKey = rand() % bucketNum;
      trustedMemory[j].x = inputTrustMemory[j];
      trustedMemory[j].key = randomKey;
      // TODO: improve
      int offset = bucketAddr1[(i + j) % bucketNum] + numRows1[(i + j) % bucketNum];
      opOneLinearScanBlock(offset * 2, (int*)(&trustedMemory[j]), (size_t)1, structureId, 1);
      std::cout<<"Assign each elem: i, j " <<i<<" " <<j<<" " << trustedMemory[j].x << " " << trustedMemory[j].key<<std::endl;
      std::cout<<"paddedSize, BUCKET_SIZE, BUCKET_NUM"<<paddedSize<<" "<<BUCKET_SIZE<<" "<<bucketNum<<std::endl;
      std::cout << arrayAddr[structureId] << arrayAddr[structureId] + 1 <<std::endl;
      paddedSize = bucketNum * BUCKET_SIZE;
      print(1);
      //Bucket_x test;
      //opOneLinearScanBlock(bucketAddr1[(i + j) % bucketNum] + numRows1[(i + j) % bucketNum], (int*)(&test), (size_t)1, structureId, 0);
      numRows1[(i + j) % bucketNum] ++;
    }
    total += std::min(BLOCK_DATA_SIZE, size - i);
  }
  free(trustedMemory);
  free(inputTrustMemory);
  
  std::cout << "=======Initial0=======\n";
  std::cout << total << std::endl;
  paddedSize = bucketNum * BUCKET_SIZE;
  std::cout<<"paddedSize: "<<paddedSize<<std::endl;
  print(structureId);
  std::cout<<sizeof(int)<<" "<<sizeof(Bucket_x)<<std::endl;
  std::cout << "=======Initial0=======\n";
  
  for (int i = 0; i < bucketNum; ++i) {
    DBGprint("currently bucket %d has %d records/%d", i, numRows1[i], BUCKET_SIZE);
    std::cout << "=======Index=======\n";
    std::cout << i <<" "<<"start index:"<< BUCKET_SIZE - numRows1[i]<<std::endl;
    std::cout << "size1: "<<(BUCKET_SIZE-numRows1[i])*sizeof(Bucket_x)<<std::endl;
    std::cout << "len: " << BUCKET_SIZE - numRows1[i] << std::endl;
    padWithDummy(structureId, bucketAddr1[i], numRows1[i]);
    print(structureId);
    std::cout << "=======Index=======\n";
  }
  
  std::cout << "=======Initial=======\n";
  paddedSize = bucketNum * BUCKET_SIZE;
  print(structureId);
  std::cout << "=======Initial=======\n";
  
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
  }
  // TODO: remove DUMMY (finished)
  for (int i = 0; i < bucketNum; ++i) {
    if (ranBinAssignIters % 2) {
      moveDummy(structureId, bucketAddr1[i]);
    } else {
      moveDummy(structureId + 1, bucketAddr2[i]);
    }
  }
  int resultId = 0;
  if (ranBinAssignIters % 2 == 1) {
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
    print();
}

void test(int structureId) {
  int pass = 1;
  int i;
  for (i = 1; i < paddedSize; i++) {
    pass &= (((Bucket_x*)arrayAddr[structureId])[i-1].x <= ((Bucket_x*)arrayAddr[structureId])[i].x);
  }

  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
    print(structureId);
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



