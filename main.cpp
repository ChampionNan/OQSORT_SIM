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


#define N 3000000//1100
#define M 2048
#define NUM_STRUCTURES 10
// #define MEM_IN_ENCLAVE 5
#define DUMMY 0xffffffff
#define NULLCHAR '\0'
// #define B 10

#define ALPHA 0.1
#define beta 0.1
#define gamma 0.1

#define BLOCK_DATA_SIZE 256
#define BUCKET_SIZE 2048//256
#define MERGE_SORT_BATCH_SIZE 256
#define WRITE_BUFFER_SIZE 256



typedef struct {
  int x;
  int key;
} Bucket_x;

bool cmpHelper(Bucket_x *a, Bucket_x *b) {
  return (a->x > b->x) ? true : false;
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
void print(int **arrayAddr, int structureId, int size);
void test(int **arrayAddr, int structureId, int size);
void callSort(int sortId, int structureId, int paddedSize, int *resId);
void smallBitonicMerge(int *a, int start, int size, int flipped);
void smallBitonicSort(int *a, int start, int size, int flipped);
void bitonicMerge(int structureId, int start, int size, int flipped, int* row1, int* row2);
void bitonicSort(int structureId, int start, int size, int flipped, int* row1, int* row2);
int greatestPowerOfTwoLessThan(int n);
void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, int* numRow2, int* outputId, int iter, int k, int* bucketAddr, int outputStructureId);
void mergeSplit(int inputStructureId, int outputStructureId, int *inputId, int *outputId, int k, int* bucketAddr, int* numRow1, int* numRow2, int iter);
void kWayMergeSort(int inputStructureId, int outputStructureId, int* numRow1, int* numRow2, int* bucketAddr, int bucketSize);
void bucketSort(int inputStructureId, int bucketId, int size, int dataStart);
int bucketOSort(int structureId, int size);
int partition(Bucket_x *arr, int low, int high);
void quickSort(Bucket_x *arr, int low, int high);
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
  // 0: OSORT, 1: bucketOSort, 2: smallBSort, 3: bitonicSort,
  int sortId = 1;

  // step1: init test numbers
  if (sortId == 2 || sortId == 3) {
    int addi = 0;
    if (N % BLOCK_DATA_SIZE != 0) {
      addi = ((N / BLOCK_DATA_SIZE) + 1) * BLOCK_DATA_SIZE - N;
    }
    X = (int*)malloc((N + addi) * sizeof(int));
    paddedSize = N + addi;
    arrayAddr[0] = X;
  } else if (sortId == 1) {
    srand((unsigned)time(NULL));
    int k = M / BLOCK_DATA_SIZE;
    assert(k >= 2 && "M/B must greater than 2");
    int bucketNum = smallestPowerOfKLargerThan(ceil(2.0 * N / BUCKET_SIZE), k);
    int bucketSize = bucketNum * BUCKET_SIZE;
    std::cout << "TOTAL BUCKET SIZE: " << bucketSize << std::endl;
    std::cout << "BUCKET NUMBER: " << bucketNum << std::endl;
    bucketx1 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
    bucketx2 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
    memset(bucketx1, 0, bucketSize*sizeof(Bucket_x));
    memset(bucketx2, 0, bucketSize*sizeof(Bucket_x));
    arrayAddr[1] = (int*)bucketx1;
    arrayAddr[2] = (int*)bucketx2;
    X = (int *) malloc(N * sizeof(int));
    arrayAddr[0] = X;
    paddedSize = N;
  } else {
    // TODO:
  }
  init(arrayAddr, 0, paddedSize);

  // step2: Create the enclave
  
  
  // step3: call sort algorithms
  start = std::chrono::high_resolution_clock::now();
  if (sortId == 2 || sortId == 3) {
    std::cout << "Test bitonic sort... " << std::endl;
    callSort(sortId, 0, paddedSize, resId);
    test(arrayAddr, 0, paddedSize);
  } else if (sortId == 1) {
    std::cout << "Test bucket oblivious sort... " << std::endl;
    callSort(sortId, 1, paddedSize, resId);
    std::cout << "Result ID: " << *resId << std::endl;
    // print(arrayAddr, *resId, paddedSize);
    test(arrayAddr, *resId, paddedSize);
  } else {
    // TODO:
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

/*
int SampleWithOutReplace(int inStructureId, int samplesId) {
  int n_prime = (int)ceil(ALPHA * N);
  int alphaM2 = (int)ceil(2 * ALPHA * M);
  int boundary = (int)ceil(N/M);
  int Msize;
  int realNum = 0; // #pivots
  int writeBacksize = 0; // #y's write back size
  int writeBackstart = 0;
  int readStart = 0;
  int *y = (int*)malloc(M * sizeof(int));
  int *trustedMemory = (int*)malloc(M * sizeof(int));
  std::random_device rd;
  std::mt19937_64 generator(rd());
  double rate = ALPHA;
  std::bernoulli_distribution b(rate);
  
  // 1. sample with bernouli
  for (int i = 0; i < boundary; i++) {
    Msize = std::min(M, N - i * M);
    opOneLinearScanBlock(readStart, trustedMemory, Msize, inStructureId, 0);
    // print(trustedMemory, Msize);
    readStart += Msize;
    for (int j = 0; j < Msize; ++j) {
      if (b(generator)) {
        n_prime --;
      } else {
        trustedMemory[j] = DUMMY;
      }
    }
    // 2. move dummy & write back to external memory
    realNum += moveDummy(trustedMemory, Msize);
    if (writeBacksize + alphaM2 >= M) {
      opOneLinearScanBlock(writeBackstart, y, writeBacksize, samplesId, 1);
      // only update here
      // print(samplesId, M);
      writeBackstart += writeBacksize;
      writeBacksize = 0;
      memcpy(y, trustedMemory, alphaM2 * sizeof(int));
    } else {
      memcpy(y + writeBacksize, trustedMemory, alphaM2 * sizeof(int));
      writeBacksize += alphaM2;
    }
  }
  opOneLinearScanBlock(writeBackstart, y, writeBacksize, samplesId, 1);
  free(trustedMemory);
  free(y);
  if (realNum < (int)(ALPHA * N)) {
    return -1;
  }
  return realNum;
}*/

// TODO: check k >= M case, change M to a smaller number
/*
int PivotsSelection(int inStructureId, int samplesId, int pivotsId) {
  // 1. sort samples
  int numSamples = -1;
  while (numSamples == -1) {
    numSamples = SampleWithOutReplace(inStructureId, samplesId);
  }
  // TODO: Error alpha
  int alpha = 0.1;
  int sampleSize = (int)ceil(2 * alpha * N);
  int *samples = (int*)malloc(sizeof(int) * sampleSize);
  opOneLinearScanBlock(0, samples, sampleSize, samplesId, 0);
  moveDummy(samples, sampleSize);
  // TODO: bitonic sort need to pad with dummy to satisfy entire block data size, so currently use quicksort
  quickSort(samples, 0, numSamples - 1);
  opOneLinearScanBlock(0, samples, numSamples, samplesId, 1);
  free(samples);
  std::cout<<"=====print samples=====\n";
  print(samplesId, numSamples);
  // 2. get pivots
  int *p = (int*)malloc(M * sizeof(int));
  double j = ALPHA * M;
  int k = 0;
  int realPivots = 0; // #pivots
  int end = (int)ceil(ALPHA * N / M);
  int *trustedMemory = (int*)malloc(M * sizeof(int));
  int writeBackstart = 0;
  int readStart = 0;
  double endPivotsIdx = ALPHA * M;
  int quitFlag = 1;
  int totalK = 0;
  // 3. pivots read & write backstd::ctype_base::alpha
  for (int i = 0; i < end && quitFlag; ++i) {
    int Msize = std::min(M, (int)ceil(ALPHA * N) - i * M);
    opOneLinearScanBlock(readStart, trustedMemory, Msize, samplesId, 0);
    readStart += Msize;
    while (j < M) {
      int indexj = (int)floor(j);
      if (k >= M) {
        opOneLinearScanBlock(writeBackstart, p, M, pivotsId, 1);
        writeBackstart += M;
        k -= M;
      }
      p[k++] = trustedMemory[indexj];
      totalK ++; // will not reduced even written back
      j += ALPHA * M;
      endPivotsIdx += ALPHA * M;
      // 4. out of samples index
      if (endPivotsIdx > numSamples - 1) {
        opOneLinearScanBlock(writeBackstart, p, k, pivotsId, 1);
        realPivots = totalK;
        quitFlag = 0;
        break;
      }
    }
    j -= M;
  }
  
  std::cout<<"-----print pivots-----\n";
  print(p, realPivots);
  std::cout<<"-----print pivots extrenal-----\n";
  // print(pivotsId, realPivots);
  free(p);
  free(trustedMemory);
  return realPivots;
}*/


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
// TODO: malloc error, need to justify return pointer != NULL
/*
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
  // 3. initialize each section
  int *offset = (int*)malloc(P * sizeof(int));
  memset(offset, 0, P * sizeof(int));
  int *writeBackOffset = (int*)malloc(P * sizeof(int));
  memset(writeBackOffset, 0, P * sizeof(int));
  int *trustedMemory = (int*)malloc(M_prime * sizeof(int));
  int *xiSec = (int*)malloc(psecSize * sizeof(int));
  memset(xiSec, DUMMY, psecSize * sizeof(int));
  int retFlag = 1;
  // 4. seperate elements
  for (int j = 0; j < end; ++j) {
    int jstart = j * psecSize;
    // TODO: error: it becomes negative number
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
  
  free(offset);
  free(writeBackOffset);
  free(trustedMemory);
  free(xiSec);
  return retFlag;
}*/

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
/*
int ObliviousSort(int inStructureId, int inSize, int sampleId, int pivotsId, int outStructureId) {
  int *trustedMemory = NULL;
  // 1. N <= M case, use quicksort
  if (N <= M) {
    trustedMemory = (int*)malloc(N);
    opOneLinearScanBlock(0, trustedMemory, N, inStructureId, 0);
    quickSort(trustedMemory, 0, N - 1);
    opOneLinearScanBlock(0, trustedMemory, N, outStructureId, 1);
    free(trustedMemory);
    return outStructureId;
  }
  // 2. select pivots
  int numPivots = -1;
  numPivots = PivotsSelection(inStructureId, sampleId, pivotsId);
  std::cout<<"=====Output Pivots=====\n";
  print(pivotsId, numPivots);
  std::cout<<"=====Output Pivots=====\n";
  // 3. Fisher-Yates shuffle
  trustedMemory = (int*)malloc(2 *  B * sizeof(int));
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
  free(trustedMemory);
  // shuffle success
  std::cout<<"-----input-----\n";
  print(inStructureId, N);
  std::cout<<"-----input-----\n";
  // 4. level iteration
  // TODO: local pointer is free but the matched memory is not free
  int r = (int)ceil(log(N / M) / log(numPivots / (1 + 2 * beta)));
  int levelSize = 0;
  for (int i = 0; i < r; ++i) {
    int jEnd = (int)ceil(pow(M/B, i));
    int W = (int)((N/M)/jEnd);
    std::cout<<"W: "<<W<<", jEnd: "<<jEnd<<std::endl;
    int *p = (int*)malloc(W * sizeof(int)); // read pivots
    double *quanP = (double*)malloc(sizeof(double) * jEnd);
    for (int j = 0; j < jEnd; ++j) {
      int wSize = std::min(W, numPivots - j * W);
      opOneLinearScanBlock(j * W, p, wSize, pivotsId, 0);
      quanP[j] = quantileCal(&p[j * W], wSize, B / M);
      std::cout<<jEnd<<"-----quantile-----\n";
      std::cout<<quanP[j]<<std::endl;
    }
    free(p);
    int XiSize = (int)ceil(N * B * (1 + 2 * beta) / M);
    // Use two part both ceiling
    levelSize = jEnd * XiSize;
    int flag = 0;
    if (i % 2 == 0) {
      while (!flag) {
        flag = DataPartition(inStructureId, outStructureId, quanP, jEnd, levelSize);
      }
    } else {
      while (!flag) {
        flag = DataPartition(outStructureId, inStructureId, quanP, jEnd, levelSize);
      }
    }
    free(quanP);
  }
  // 5. sort last level
  int jEnd = (int)pow(M/B, r);
  int blockSize = (int)ceil(M/B);
  int totalReal = 0; // use for write back address
  trustedMemory = (int*)malloc(blockSize * sizeof(int));
  for (int j = 0; j < jEnd; ++j) {
    int readSize = std::min(blockSize, levelSize - j * blockSize);
    if (r % 2 == 0) {
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
  free(trustedMemory);
  return r % 2 == 0; // return 1->outId; 0->inId
}*/


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
  // bitonic sort
  if (sortId == 1) {
     *resId = bucketOSort(structureId, paddedSize);
  }
  if (sortId == 3) {
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
  Bucket_x *junk = (Bucket_x*)malloc(len * sizeof(Bucket_x));

  for (int i = 0; i < len; ++i) {
    junk[i].x = -1;
    junk[i].key = -1;
  }
  
  opOneLinearScanBlock(2 * (start + realNum), (int*)junk, len, structureId, 1);
  free(junk);
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

void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, int* numRow2, int* outputId, int iter, int k, int* bucketAddr, int outputStructureId) {
  int batchSize = 256; // 8192
  // Bucket_x *buf0 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
  // Bucket_x *buf1 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
  // TODO: FREE these malloc
  Bucket_x **buf = (Bucket_x**)malloc(k * sizeof(Bucket_x*));
  for (int i = 0; i < k; ++i) {
    buf[i] = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
  }
  
  // int counter0 = 0, counter1 = 0;
  int randomKey = 0;
  int *counter = (int*)malloc(k * sizeof(int));
  memset(counter, 0, k * sizeof(int));
  
  for (int i = 0; i < inputBufferLen; ++i) {
    if ((inputBuffer[i].key != DUMMY) && (inputBuffer[i].x != DUMMY)) {
      randomKey = inputBuffer[i].key;
      for (int j = 0; j < k; ++j) {
        if (isTargetIterK(randomKey, iter, k, j)) {
          buf[j][counter[j] % batchSize] = inputBuffer[i];
          counter[j]++;
          // std::cout << "couter j: " << counter[j] << std::endl;
          if (counter[j] % batchSize == 0) {
            opOneLinearScanBlock(2 * (bucketAddr[outputId[j]] +  numRow2[outputId[j]]), (int*)buf[j], (size_t)batchSize, outputStructureId, 1);
            numRow2[outputId[j]] += batchSize;
          }
        }
      }
    }
  }
  
  for (int j = 0; j < k; ++j) {
    opOneLinearScanBlock(2 * (bucketAddr[outputId[j]] + numRow2[outputId[j]]), (int*)buf[j], (size_t)(counter[j] % batchSize), outputStructureId, 1);
    numRow2[outputId[j]] += counter[j] % batchSize;
    free(buf[j]);
  }
  free(counter);
}

void mergeSplit(int inputStructureId, int outputStructureId, int *inputId, int *outputId, int k, int* bucketAddr, int* numRow1, int* numRow2, int iter) {
  Bucket_x *inputBuffer = (Bucket_x*)malloc(sizeof(Bucket_x) * BUCKET_SIZE);
  // BLOCKs
  for (int i = 0; i < k; ++i) {
    opOneLinearScanBlock(2 * bucketAddr[inputId[i]], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
    mergeSplitHelper(inputBuffer, numRow1[inputId[i]], numRow2, outputId, iter, k, bucketAddr, outputStructureId);
    for (int j = 0; j < k; ++j) {
      if (numRow2[outputId[j]] > BUCKET_SIZE) {
        printf("overflow error during merge split!\n");
      }
    }
  }
  
  for (int j = 0; j < k; ++j) {
    padWithDummy(outputStructureId, bucketAddr[outputId[j]], numRow2[outputId[j]]);
  }
  
  free(inputBuffer);
}

void kWayMergeSort(int inputStructureId, int outputStructureId, int* numRow1, int* numRow2, int* bucketAddr, int bucketSize) {
  int mergeSortBatchSize = (int)MERGE_SORT_BATCH_SIZE; // 256
  int writeBufferSize = (int)WRITE_BUFFER_SIZE; // 8192
  int numWays = bucketSize;
  HeapNode inputHeapNodeArr[numWays];
  int totalCounter = 0;
  
  int *readBucketAddr = (int*)malloc(sizeof(int) * numWays);
  memcpy(readBucketAddr, bucketAddr, sizeof(int) * numWays);
  int writeBucketAddr = 0;
  
  for (int i = 0; i < numWays; ++i) {
    HeapNode node;
    node.data = (Bucket_x*)malloc(mergeSortBatchSize * sizeof(Bucket_x));
    node.bucketIdx = i;
    node.elemIdx = 0;
    opOneLinearScanBlock(2 * readBucketAddr[i], (int*)node.data, (size_t)std::min(mergeSortBatchSize, numRow1[i]), inputStructureId, 0);
    inputHeapNodeArr[i] = node;
    readBucketAddr[i] += std::min(mergeSortBatchSize, numRow1[i]);
  }
  
  Heap heap(inputHeapNodeArr, numWays, mergeSortBatchSize);
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
      numRow2[temp->bucketIdx] += writeBufferSize;
      writeBufferCounter = 0;
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
  numRow2[0] += writeBufferCounter;
  free(writeBuffer);
  free(readBucketAddr);
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
  int k = M / BLOCK_DATA_SIZE;
  int bucketNum = smallestPowerOfKLargerThan(ceil(2.0 * size / BUCKET_SIZE), k);
  int ranBinAssignIters = log(bucketNum)/log(k) - 1;
  std::cout << "Iteration times: " << ranBinAssignIters << std::endl;
  // srand((unsigned)time(NULL));
  // std::cout << "Iters:" << ranBinAssignIters << std::endl;
  int *bucketAddr = (int*)malloc(bucketNum * sizeof(int));
  int *numRow1 = (int*)malloc(bucketNum * sizeof(int));
  int *numRow2 = (int*)malloc(bucketNum * sizeof(int));
  memset(numRow1, 0, bucketNum * sizeof(int));
  memset(numRow2, 0, bucketNum * sizeof(int));
  
  for (int i = 0; i < bucketNum; ++i) {
    bucketAddr[i] = i * BUCKET_SIZE;
  }
  
  Bucket_x *trustedMemory = (Bucket_x*)malloc(BLOCK_DATA_SIZE * sizeof(Bucket_x));
  int *inputTrustMemory = (int*)malloc(BLOCK_DATA_SIZE * sizeof(int));
  int total = 0;
  int offset;

  for (int i = 0; i < size; i += BLOCK_DATA_SIZE) {
    opOneLinearScanBlock(i, inputTrustMemory, std::min(BLOCK_DATA_SIZE, size - i), structureId - 1, 0);
    int randomKey;
    for (int j = 0; j < std::min(BLOCK_DATA_SIZE, size - i); ++j) {
      // oe_random(&randomKey, 4);
      randomKey = (int)rand();
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
    // DBGprint("currently bucket %d has %d records/%d", i, numRow1[i], BUCKET_SIZE);
    padWithDummy(structureId, bucketAddr[i], numRow1[i]);
  }
  // print(arrayAddr, structureId, bucketNum * BUCKET_SIZE);
  // std::cout << "Iters:" << ranBinAssignIters << std::endl;
  // std::cout << "k:" << k << std::endl;
  int *inputId = (int*)malloc(k * sizeof(int));
  int *outputId = (int*)malloc(k *sizeof(int));
  
  for (int i = 0; i < ranBinAssignIters; ++i) {
    if (i % 2 == 0) {
      for (int j = 0; j < bucketNum / k; ++j) {
        int jj = (j / (int)pow(k, i)) * (int)pow(k, i);
        for (int m = 0; m < k; ++m) {
          inputId[m] = j + jj + m * (int)pow(k, i);
          outputId[m] = k * j + m;
          // std::cout << inputId[m] << ", " << outputId[m] << std::endl;
        }
        
        mergeSplit(structureId, structureId + 1, inputId, outputId, k, bucketAddr, numRow1, numRow2, i);
      }
      int count = 0;
      for (int k = 0; k < bucketNum; ++k) {
        numRow1[k] = 0;
        count += numRow2[k];
      }
      printf("after %dth merge split, we have %d tuples\n", i, count);
    } else {
      for (int j = 0; j < bucketNum / k; ++j) {
        int jj = (j / (int)pow(k, i)) * (int)pow(k, i);
        for (int m = 0; m < k; ++m) {
          inputId[m] = j + jj + m * (int)pow(k, i);
          outputId[m] = k * j + m;
        }
        mergeSplit(structureId + 1, structureId, inputId, outputId, k, bucketAddr, numRow2, numRow1, i);
      }
      int count = 0;
      for (int k = 0; k < bucketNum; ++k) {
        numRow2[k] = 0;
        count += numRow1[k];
      }
      printf("after %dth merge split, we have %d tuples\n", i, count);
    }
    printf("\n\n Finish random bin assignment iter%dth out of %d\n\n", i, ranBinAssignIters);
  }
  
  int resultId = 0;
  if (ranBinAssignIters % 2 == 0) {
    for (int i = 0; i < bucketNum; ++i) {
      bucketSort(structureId, i, numRow1[i], bucketAddr[i]);
    }
    kWayMergeSort(structureId, structureId + 1, numRow1, numRow2, bucketAddr, bucketNum);
    
    resultId = structureId + 1;
  } else {
    for (int i = 0; i < bucketNum; ++i) {
      bucketSort(structureId + 1, i, numRow2[i], bucketAddr[i]);
    }
    kWayMergeSort(structureId + 1, structureId, numRow2, numRow1, bucketAddr, bucketNum);
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

/** -------------- SUB-PROCEDURES  ----------------- **/

/** procedure test() : verify sort results **/
void init(int **arrayAddr, int structurId, int size) {
  int i;
  int *addr = (int*)arrayAddr[structurId];
  for (i = 0; i < size; i++) {
    addr[i] = (size - i);
  }
}


void print(int **arrayAddr, int structureId, int size) {
  int i;
  if(structureSize[structureId] == 4) {
    int *addr = (int*)arrayAddr[structureId];
    for (i = 0; i < size; i++) {
      printf("%d ", addr[i]);
    }
  } else if (structureSize[structureId] == 8) {
    Bucket_x *addr = (Bucket_x*)arrayAddr[structureId];
    for (i = 0; i < size; i++) {
      printf("(%d, %d) ", addr[i].x, addr[i].key);
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



