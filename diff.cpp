diff --git a/main.cpp b/main.cpp
index 684efcf..6f12603 100644
--- a/main.cpp
+++ b/main.cpp
@@ -55,8 +55,8 @@
 #include <chrono>
 
 
-#define N 200//1100
-#define M 128
+#define N 2000//1100
+#define M 512
 #define NUM_STRUCTURES 10
 #define MEM_IN_ENCLAVE 5
 #define DUMMY 0xffffffff
@@ -68,7 +68,7 @@
 #define gamma 0.1
 
 #define BLOCK_DATA_SIZE 256
-#define BUCKET_SIZE 10000//256
+#define BUCKET_SIZE 256//256
 #define MERGE_SORT_BATCH_SIZE 256
 #define WRITE_BUFFER_SIZE 256
 
@@ -177,12 +177,11 @@ void Heap::replaceRoot(HeapNode x) {
 
 int printf(const char *fmt, ...);
 int greatestPowerOfTwoLessThan(int n);
-int smallestPowerOfTwoLargerThan(int n);
+int smallestPowerOfKLargerThan(int n, int k);
 void opOneLinearScanBlock(int index, int* block, size_t blockSize, int structureId, int write);
 void padWithDummy(int structureId, int start, int realNum);
-bool isTargetBitOne(int randomKey, int iter);
+bool isTargetIterK(int randomKey, int iter, int k, int num);
 void swapRow(Bucket_x *a, Bucket_x *b);
-int smallestPowerOfTwoLargerThan(int n);
 void init(int **arrayAddr, int structurId, int size);
 void print(int **arrayAddr, int structureId, int size);
 void test(int **arrayAddr, int structureId, int size);
@@ -192,8 +191,8 @@ void smallBitonicSort(int *a, int start, int size, int flipped);
 void bitonicMerge(int structureId, int start, int size, int flipped, int* row1, int* row2);
 void bitonicSort(int structureId, int start, int size, int flipped, int* row1, int* row2);
 int greatestPowerOfTwoLessThan(int n);
-void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, int* numRow2, int outputId0, int outputId1, int iter, int* bucketAddr, int outputStructureId);
-void mergeSplit(int inputStructureId, int outputStructureId, int inputId0, int inputId1, int outputId0, int outputId1, int* bucketAddr, int* numRow1, int* numRow2, int iter);
+void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, int* numRow2, int* outputId, int iter, int k, int* bucketAddr, int outputStructureId);
+void mergeSplit(int inputStructureId, int outputStructureId, int *inputId, int *outputId, int k, int* bucketAddr, int* numRow1, int* numRow2, int iter);
 void kWayMergeSort(int inputStructureId, int outputStructureId, int* numRow1, int* numRow2, int* bucketAddr, int bucketSize);
 void bucketSort(int inputStructureId, int bucketId, int size, int dataStart);
 int bucketOSort(int structureId, int size);
@@ -264,9 +263,12 @@ int main(int argc, const char* argv[]) {
     arrayAddr[0] = X;
   } else if (sortId == 1) {
     srand((unsigned)time(NULL));
-    int bucketNum = smallestPowerOfTwoLargerThan(ceil(2.0 * N / BUCKET_SIZE));
+    int k = M / BLOCK_DATA_SIZE;
+    assert(k >= 2 && "M/B must greater than 2");
+    int bucketNum = smallestPowerOfKLargerThan(ceil(2.0 * N / BUCKET_SIZE), k);
     int bucketSize = bucketNum * BUCKET_SIZE;
     std::cout << "TOTAL BUCKET SIZE: " << bucketSize << std::endl;
+    std::cout << "BUCKET NUMBER: " << bucketNum << std::endl;
     bucketx1 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
     bucketx2 = (Bucket_x*)malloc(bucketSize * sizeof(Bucket_x));
     memset(bucketx1, 0, bucketSize*sizeof(Bucket_x));
@@ -294,7 +296,7 @@ int main(int argc, const char* argv[]) {
     std::cout << "Test bucket oblivious sort: " << std::endl;
     callSort(sortId, 1, paddedSize, resId);
     std::cout << "Result ID: " << *resId << std::endl;
-    print(arrayAddr, *resId, paddedSize);
+    // print(arrayAddr, *resId, paddedSize);
     test(arrayAddr, *resId, paddedSize);
   } else {
     // TODO:
@@ -325,12 +327,12 @@ int greatestPowerOfTwoLessThan(int n) {
     return k >> 1;
 }
 
-int smallestPowerOfTwoLargerThan(int n) {
-  int k = 1;
-  while (k > 0 && k < n) {
-    k = k << 1;
+int smallestPowerOfKLargerThan(int n, int k) {
+  int num = 1;
+  while (num > 0 && num < n) {
+    num = num * k;
   }
-  return k;
+  return num;
 }
 
 
@@ -824,71 +826,70 @@ int moveDummy(int *a, int size) {
   return k;
 }*/
 
-bool isTargetBitOne(int randomKey, int iter) {
-  assert(iter >= 1);
-  return (randomKey & (0x01 << (iter - 1))) == 0 ? false : true;
+bool isTargetIterK(int randomKey, int iter, int k, int num) {
+  while (iter) {
+    randomKey = randomKey / k;
+    iter--;
+  }
+  // return (randomKey & (0x01 << (iter - 1))) == 0 ? false : true;
+  return (randomKey % k) == num;
 }
 
-void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, int* numRow2, int outputId0, int outputId1, int iter, int* bucketAddr, int outputStructureId) {
+void mergeSplitHelper(Bucket_x *inputBuffer, int inputBufferLen, int* numRow2, int* outputId, int iter, int k, int* bucketAddr, int outputStructureId) {
   int batchSize = 256; // 8192
-  Bucket_x *buf0 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
-  Bucket_x *buf1 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
-  int counter0 = 0, counter1 = 0;
+  // Bucket_x *buf0 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
+  // Bucket_x *buf1 = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
+  // TODO: FREE these malloc
+  Bucket_x **buf = (Bucket_x**)malloc(k * sizeof(Bucket_x*));
+  for (int i = 0; i < k; ++i) {
+    buf[i] = (Bucket_x*)malloc(batchSize * sizeof(Bucket_x));
+  }
+  
+  // int counter0 = 0, counter1 = 0;
   int randomKey = 0;
+  int *counter = (int*)malloc(k * sizeof(int));
+  memset(counter, 0, k * sizeof(int));
   
   for (int i = 0; i < inputBufferLen; ++i) {
     if ((inputBuffer[i].key != DUMMY) && (inputBuffer[i].x != DUMMY)) {
       randomKey = inputBuffer[i].key;
-      
-      if (isTargetBitOne(randomKey, iter + 1)) {
-        buf1[counter1 % batchSize] = inputBuffer[i];
-        counter1 ++;
-        if (counter1 % batchSize == 0) {
-          opOneLinearScanBlock(2 * (bucketAddr[outputId1] +  numRow2[outputId1]), (int*)buf1, (size_t)batchSize, outputStructureId, 1);
-          numRow2[outputId1] += batchSize;
-          memset(buf1, NULLCHAR, batchSize * sizeof(Bucket_x));
-        }
-      } else {
-        buf0[counter0 % batchSize] = inputBuffer[i];
-        counter0 ++;
-        if (counter0 % batchSize == 0) {
-          opOneLinearScanBlock(2 * (bucketAddr[outputId0] + numRow2[outputId0]), (int*)buf0, (size_t)batchSize, outputStructureId, 1);
-          numRow2[outputId0] += batchSize;
-          memset(buf0, NULLCHAR, batchSize * sizeof(Bucket_x));
+      for (int j = 0; j < k; ++j) {
+        if (isTargetIterK(randomKey, iter, k, j)) {
+          buf[j][counter[j] % batchSize] = inputBuffer[i];
+          counter[j]++;
+          if (counter[j] % batchSize == 0) {
+            opOneLinearScanBlock(2 * (bucketAddr[outputId[j]] +  numRow2[outputId[j]]), (int*)buf[j], (size_t)batchSize, outputStructureId, 1);
+            numRow2[outputId[j]] += batchSize;
+          }
         }
       }
     }
   }
   
-  opOneLinearScanBlock(2 * (bucketAddr[outputId1] + numRow2[outputId1]), (int*)buf1, (size_t)(counter1 % batchSize), outputStructureId, 1);
-  numRow2[outputId1] += counter1 % batchSize;
-  opOneLinearScanBlock(2 * (bucketAddr[outputId0] + numRow2[outputId0]), (int*)buf0, (size_t)(counter0 % batchSize), outputStructureId, 1);
-  numRow2[outputId0] += counter0 % batchSize;
-  
-  free(buf0);
-  free(buf1);
+  for (int j = 0; j < k; ++j) {
+    opOneLinearScanBlock(2 * (bucketAddr[outputId[j]] + numRow2[outputId[j]]), (int*)buf[j], (size_t)(counter[j] % batchSize), outputStructureId, 1);
+    free(buf[j]);
+  }
+  free(counter);
 }
 
-void mergeSplit(int inputStructureId, int outputStructureId, int inputId0, int inputId1, int outputId0, int outputId1, int* bucketAddr, int* numRow1, int* numRow2, int iter) {
+void mergeSplit(int inputStructureId, int outputStructureId, int *inputId, int *outputId, int k, int* bucketAddr, int* numRow1, int* numRow2, int iter) {
   Bucket_x *inputBuffer = (Bucket_x*)malloc(sizeof(Bucket_x) * BUCKET_SIZE);
-  // BLOCK#0
-  opOneLinearScanBlock(2 * bucketAddr[inputId0], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
-  mergeSplitHelper(inputBuffer, numRow1[inputId0], numRow2, outputId0, outputId1, iter, bucketAddr, outputStructureId);
-  if (numRow2[outputId0] > BUCKET_SIZE || numRow2[outputId1] > BUCKET_SIZE) {
-    // DBGprint("overflow error during merge split!\n");
+  // BLOCKs
+  for (int i = 0; i < k; ++i) {
+    opOneLinearScanBlock(2 * bucketAddr[inputId[i]], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
+    mergeSplitHelper(inputBuffer, numRow1[inputId[i]], numRow2, outputId, iter, k, bucketAddr, outputStructureId);
+    for (int j = 0; j < k; ++j) {
+      if (numRow2[outputId[j]] > BUCKET_SIZE) {
+        printf("overflow error during merge split!\n");
+      }
+    }
   }
   
-  // BLOCK#1
-  opOneLinearScanBlock(2 * bucketAddr[inputId1], (int*)inputBuffer, BUCKET_SIZE, inputStructureId, 0);
-  mergeSplitHelper(inputBuffer, numRow1[inputId1], numRow2, outputId0, outputId1, iter, bucketAddr, outputStructureId);
-  
-  if (numRow2[outputId0] > BUCKET_SIZE || numRow2[outputId1] > BUCKET_SIZE) {
-    // DBGprint("overflow error during merge split!\n");
+  for (int j = 0; j < k; ++j) {
+    padWithDummy(outputStructureId, bucketAddr[outputId[j]], numRow2[outputId[j]]);
   }
   
-  padWithDummy(outputStructureId, bucketAddr[outputId1], numRow2[outputId1]);
-  padWithDummy(outputStructureId, bucketAddr[outputId0], numRow2[outputId0]);
-  
   free(inputBuffer);
 }
 
@@ -962,10 +963,11 @@ void bucketSort(int inputStructureId, int bucketId, int size, int dataStart) {
 
 // int inputTrustMemory[BLOCK_DATA_SIZE];
 int bucketOSort(int structureId, int size) {
-  int bucketNum = smallestPowerOfTwoLargerThan(ceil(2.0 * size / BUCKET_SIZE));
-  int ranBinAssignIters = log2(bucketNum) - 1;
+  int k = M / BLOCK_DATA_SIZE;
+  int bucketNum = smallestPowerOfKLargerThan(ceil(2.0 * size / BUCKET_SIZE), k);
+  int ranBinAssignIters = log(bucketNum)/log(k) - 1;
   // srand((unsigned)time(NULL));
-
+  // std::cout << "Iters:" << ranBinAssignIters << std::endl;
   int *bucketAddr = (int*)malloc(bucketNum * sizeof(int));
   int *numRow1 = (int*)malloc(bucketNum * sizeof(int));
   int *numRow2 = (int*)malloc(bucketNum * sizeof(int));
@@ -1003,39 +1005,51 @@ int bucketOSort(int structureId, int size) {
     // DBGprint("currently bucket %d has %d records/%d", i, numRow1[i], BUCKET_SIZE);
     padWithDummy(structureId, bucketAddr[i], numRow1[i]);
   }
-
+  // print(arrayAddr, structureId, bucketNum * BUCKET_SIZE);
+  std::cout << "Iters:" << ranBinAssignIters << std::endl;
+  std::cout << "k:" << k << std::endl;
+  int *inputId = (int*)malloc(k * sizeof(int));
+  int *outputId = (int*)malloc(k *sizeof(int));
+  
   for (int i = 0; i < ranBinAssignIters; ++i) {
     if (i % 2 == 0) {
-      for (int j = 0; j < bucketNum / 2; ++j) {
-        int jj = (j / (int)pow(2, i)) * (int)pow(2, i);
-        mergeSplit(structureId, structureId + 1, j + jj, j + jj + (int)pow(2, i), 2 * j, 2 * j + 1, bucketAddr, numRow1, numRow2, i);
+      for (int j = 0; j < bucketNum / k; ++j) {
+        int jj = (j / (int)pow(k, i)) * (int)pow(k, i);
+        for (int m = 0; m < k; ++m) {
+          inputId[m] = j + jj + m * (int)pow(k, i);
+          outputId[m] = k * j + m;
+        }
+        mergeSplit(structureId, structureId + 1, inputId, outputId, k, bucketAddr, numRow1, numRow2, i);
       }
       int count = 0;
       for (int k = 0; k < bucketNum; ++k) {
         numRow1[k] = 0;
         count += numRow2[k];
       }
-      // DBGprint("after %dth merge split, we have %d tuples\n", i, count);
+      printf("after %dth merge split, we have %d tuples\n", i, count);
     } else {
-      for (int j = 0; j < bucketNum / 2; ++j) {
-        int jj = (j / (int)pow(2, i)) * (int)pow(2, i);
-        mergeSplit(structureId + 1, structureId, j + jj, j + jj + (int)pow(2, i), 2 * j, 2 * j + 1, bucketAddr, numRow2, numRow1, i);
+      for (int j = 0; j < bucketNum / k; ++j) {
+        int jj = (j / (int)pow(k, i)) * (int)pow(k, i);
+        for (int m = 0; m < k; ++m) {
+          inputId[m] = j + jj + m * (int)pow(k, i);
+          outputId[m] = k * j + m;
+        }
+        mergeSplit(structureId + 1, structureId, inputId, outputId, k, bucketAddr, numRow2, numRow1, i);
       }
       int count = 0;
       for (int k = 0; k < bucketNum; ++k) {
         numRow2[k] = 0;
         count += numRow1[k];
       }
-      // DBGprint("after %dth merge split, we have %d tuples\n", i, count);
+      printf("after %dth merge split, we have %d tuples\n", i, count);
     }
-    // DBGprint("\n\n Finish random bin assignment iter%dth out of %d\n\n", i, ranBinAssignIters);
+    printf("\n\n Finish random bin assignment iter%dth out of %d\n\n", i, ranBinAssignIters);
   }
   
   int resultId = 0;
   if (ranBinAssignIters % 2 == 0) {
     for (int i = 0; i < bucketNum; ++i) {
       bucketSort(structureId, i, numRow1[i], bucketAddr[i]);
-      print(arrayAddr, structureId, N);
     }
     kWayMergeSort(structureId, structureId + 1, numRow1, numRow2, bucketAddr, bucketNum);
     
@@ -1043,13 +1057,12 @@ int bucketOSort(int structureId, int size) {
   } else {
     for (int i = 0; i < bucketNum; ++i) {
       bucketSort(structureId + 1, i, numRow2[i], bucketAddr[i]);
-      print(arrayAddr, structureId + 1, N);
     }
     kWayMergeSort(structureId + 1, structureId, numRow2, numRow1, bucketAddr, bucketNum);
     resultId = structureId;
   }
   test(arrayAddr, resultId, N);
-  print(arrayAddr, resultId, N);
+  // print(arrayAddr, resultId, N);
   free(bucketAddr);
   free(numRow1);
   free(numRow2);
