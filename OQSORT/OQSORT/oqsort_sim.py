# import numpy as np
import random
import math
import time
import sympy
from scipy.stats import hypergeom


class SortBase:
    """
    OCALLS & SUPPORT
    """
    def __init__(self, N, M, B):
        # params
        self.N, self.M = N, M
        self.B = B
        self.DUMMY = 0xffffffff
        # inStructureId, sampleId, outStructureId1, outStructureId2
        self.data = [[], [], [], []]
        # IO cost counting
        self.IOcost = 0

    def OcallReadBlock(self, index, blockSize, structureId):
        ret = self.data[structureId][index:index+blockSize]
        self.IOcost += math.ceil(blockSize/self.B)
        return ret

    def OcallWriteBlock(self, index, buffer, blockSize, structureId):
        self.data[structureId][index:index+blockSize] = buffer
        self.IOcost += math.ceil(blockSize/self.B)

    # TODO: check all this function calling
    def opOneLinearScanBlock(self, index, block, blockSize, structureId, write=0, dummyNum=0):
        if not (blockSize + dummyNum):
            return []
        if not write:
            total = self.OcallReadBlock(index, blockSize, structureId)
            return total
        else:
            self.OcallWriteBlock(index, block + [self.DUMMY] * dummyNum, blockSize + dummyNum, structureId)

    def padWithDummy(self, structureId, start, realNum, secSize):
        lenth = secSize - realNum
        if lenth <= 0:
            return
        junk = [self.DUMMY] * lenth
        self.opOneLinearScanBlock(start + realNum, junk, lenth, structureId, 1)

    def moveDummy(self, a, size):
        k = 0
        for i in range(size):
            if a[i] != self.DUMMY:
                if i != k:
                    a[i], a[k] = a[k], a[i]
                    k += 1
                else:
                    k += 1
        return k

    def init(self, structureId, size):
        for i in range(size):
            self.data[structureId].append(size - i)
        random.shuffle(self.data[structureId])

    def test(self, structureId, size):
        passFlag = True
        for i in range(1, size):
            if self.data[structureId][i - 1] > self.data[structureId][i]:
                passFlag = False
                break
        if passFlag:
            print("TEST Passed")
        else:
            print("Test Failed")

    def testWithDummy(self, structureId, size):
        i, j = 0, 0
        for i in range(size):
            if self.data[structureId][i] != self.DUMMY:
                break
        while i < size and j < size:
            findFlag = False
            for j in range(i + 1, size):
                if self.data[structureId][j] != self.DUMMY:
                    findFlag = True
                    break

            if findFlag and self.data[structureId][i] <= self.data[structureId][j]:
                i = j
            elif not findFlag:
                print("TEST Passed")
                return
            else:
                print("TEST Failed")
        return


class OQSORT(SortBase):
    """
    implementation of OQSORT
    """
    def __init__(self, N, M, B, structureId, paddedSize):
        super(OQSORT, self).__init__(N, M, B)
        self.structureId = structureId
        self.paddedSize = paddedSize
        self.resId = -1
        self.resN = self.N
        self.sortId = -1
        self.is_tight = -1
        # TODO: OQSORT params
        self.ALPHA, self.BETA, self.P, self.IdealCost = -1, -1, -1, -1
        # used for 2 level, 2nd sample params
        self._alpha, self._beta, self._p, self._cost, self._is_tight = -1, -1, -1, -1, -1
        # Memory Load partition index array
        self.partitionIdx = []

    def onelevel(self, is_tight, kappa=27.8):
        x = sympy.Symbol('x')
        g = x ** 2 * (1 - x) / (1 + x) ** 2 / (1 + x / 2) / (1 + 2 * x)
        y = g - 2 * (1 + 2 * x) * self.N * self.B / self.M / self.M * (kappa + 1 + 2 * math.log(self.N / self.M))
        res = sympy.solveset(y, x, sympy.Interval(0, 1))
        if len(res) == 0:
            raise ValueError("N too large!")
        beta = min(res)
        alpha = (kappa + 1 + math.log(self.N)) * 4 * (1 + beta) * (1 + 2 * beta) / beta / beta / self.M
        if alpha * self.N > self.M - self.B:
            raise ValueError("N too large!")
        p = math.ceil((1 + 2 * beta) * self.N / self.M)
        print("alpha=%f, beta=%f, p=%d" % (alpha, beta, p))
        if is_tight:
            cost = 7 + 4 * beta
        else:
            cost = 6 + 6 * beta + alpha * self.B
        self.ALPHA, self.BETA, self.P, self.IdealCost = alpha, beta, p, cost
        self.sortId = not is_tight
        self.is_tight = is_tight

    # TODO: Two sets of alpha, beta, p
    def twolevel(self, is_tight, kappa=27.8):
        x = sympy.Symbol('x')
        g = x ** 2 * (1 - x) / (1 + x) ** 2 / (1 + x / 2) / (1 + 2 * x)
        y = g ** 2 - 4 * self.B * self.B * (1 + 2 * x) * self.N / self.M ** 3 * (
                kappa + 2 + 1.5 * math.log(self.N / self.M)) ** 2
        res = sympy.solveset(y, x, sympy.Interval(0, 1))
        if len(res) == 0:
            raise ValueError("N too large!")
        beta = min(res)
        alpha = (kappa + 2 + math.log(self.N)) * 4 * (1 + beta) * (1 + 2 * beta) / beta / beta / self.M
        p = math.ceil(math.sqrt((1 + 2 * beta) * self.N / self.M))
        _alpha, _beta, _p, _cost = self.onelevel(False, kappa + 1)
        print("alpha=%f, beta=%f, p=%d" % (_alpha, _beta, _p))
        if is_tight:
            cost = 9 + 7 * alpha + 8 * beta
        else:
            cost = 8 + (6 + self.B) * alpha + 10 * beta
        self.ALPHA, self.BETA, self.P, self.IdealCost = alpha, beta, p, cost
        self.sortId = not is_tight
        self.is_tight = is_tight

    def call(self):
        """
        Method to Call OQSORT
        :return:
        """
        if not self.sortId:
            self.resId = self.ObliviousTightSort(self.structureId, self.paddedSize, self.structureId + 1,
                                                 self.structureId + 2)
            self.test(self.resId, self.paddedSize)
        else:
            self.resId, self.resN = self.ObliviousLooseSort(self.structureId, self.paddedSize, self.structureId + 1,
                                                            self.structureId + 2)
            self.testWithDummy(self.resId, self.resN)

        f = open("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/OQSORT/OQSORT/outputpy.txt", "w")
        for idx in range(self.resN):
            f.write(str(self.data[self.resId][idx]) + str(' '))
            if not (idx % 10):
                f.write('\n')

        f.close()

    def call2(self):
        """
        Method to Call Two-Level OQSORT
        :return:
        """
        if not self.sortId:
            self.resId = self.ObliviousTightSort2(self.structureId, self.paddedSize, self.structureId + 1,
                                                 self.structureId + 2)
            self.test(self.resId, self.paddedSize)
        else:
            self.resId, self.resN = self.ObliviousLooseSort2(self.structureId, self.paddedSize, self.structureId + 1,
                                                            self.structureId + 2)
            self.testWithDummy(self.resId, self.resN)

        print("Ideal Cost, Actual Cost, Sample Cost, Partition Cost, Final Cost: ")
        print(str(self.IdealCost / self.N * self.B) + str(', ') +
              str(self.IOcost / self.N * self.B) + str(', ') +
              str(self.sampleCost / self.N * self.B) + str(', ') +
              str(self.partitionCost / self.N * self.B) + str(', ') +
              str(self.finalCost / self.N * self.B) + str(', '))
        f = open("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/OQSORT/OQSORT/outputpy.txt", "w")
        for idx, num in enumerate(self.data[self.resId]):
            f.write(str(num) + str(' '))
            if not (idx % 10):
                f.write('\n')
        f.close()

    def Hypergeometric(self, NN, Msize, n_prime):
        m = 0
        for j in range(Msize):
            if random.random() < n_prime / NN:
                m += 1
                n_prime -= 1
            NN -= 1
        return m

    def Sample(self, inStructureId, trustedM2, is_tight, is_rec=0):
        N_prime = self.N
        alpha = self.ALPHA if not is_rec else self._alpha
        n_prime = math.ceil(alpha * self.N)
        boundary = math.ceil(self.N / self.B)
        realNum = 0
        readStart = 0
        trustedM1 = []

        for i in range(boundary):
            Msize = min(self.B, self.N - i * self.B)
            m = self.Hypergeometric(N_prime, Msize, n_prime)
            # m = hypergeom.rvs(N_prime, n_prime, Msize, size=1)[0]
            if is_tight or (not is_tight and m > 0):
                trustedM1 = self.opOneLinearScanBlock(readStart, trustedM1, Msize, inStructureId, 0)
                readStart += Msize
                # random.shuffle(trustedM1)
                # trustedM2.extend(trustedM1[0:m])
                trustedM2.extend(random.sample(trustedM1, m))
                realNum += m
                n_prime -= m
            N_prime -= Msize

        trustedM2.sort()
        print(str(realNum) + ', ' + str(alpha * self.N))
        return realNum

    def SampleRecT(self, inStructureId, is_tight, sampleId):
        """
        Finish sample recursive selection
        """
        N_prime = N
        n_prime = math.ceil(self.ALPHA * self.N)
        boundary = math.ceil(self.N / self.M)
        realNum = 0
        readStart = 0
        writeBackstart = 0
        trustedM1 = []

        for i in range(boundary):
            Msize = min(self.M, self.N - i * self.M)
            alphaM22 = math.ceil(2 * self.ALPHA * Msize)
            trustedM1 = self.opOneLinearScanBlock(readStart, trustedM1, Msize, inStructureId, 0)
            readStart += Msize
            m = self.Hypergeometric(N_prime, Msize, n_prime)
            realNum += m
            random.shuffle(trustedM1)
            trustedM1[m:Msize] = [self.DUMMY] * (Msize - m)
            self.opOneLinearScanBlock(writeBackstsrt, trustedM1[0:alphaM22], alphaM22, sampleId, 1)
            writeBackstart += alphaM22
            N_prime -= Msize
            n_prime -= m
            if n_prime <= 0:
                break
        # TODO: contains DUMMY
        if writeBackstart > self.M:
            pivots = ObliviousLooseSortRec(sampleId, writeBackstart, )

    def quantileCal(self, samples, start, end, p):
        sampleSize = end - start
        for i in range(1, p):
            samples[i] = samples[i * sampleSize // p]
        samples[0] = float('-inf')
        samples[p] = float('inf')
        return samples[0:p + 1]

    def partition(self, arr, low, high, pivot):
        """
        Return partition border index
        """
        i = low - 1
        for j in range(low, high+1):
            if pivot > arr[j]:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        return i

    def quickSort(self, arr, low, high, pivots, left, right):
        """
        low, high represents pivots' index
        """
        # use pivots to judge the end
        if right >= left:
            pivotIdx = (left + right) >> 1
            pivot = pivots[pivotIdx]
            mid = self.partition(arr, low, high, pivot)
            self.partitionIdx.append(mid)
            self.quickSort(arr, low, mid, pivots, left, pivotIdx - 1)
            self.quickSort(arr, mid + 1, high, pivots, pivotIdx + 1, right)

    def checkSection(self, trustedM1, trustedM3):
        passFlag = False
        for j in range(len(self.partitionIdx)-2):
            index1, index2, index3 = self.partitionIdx[j], self.partitionIdx[j+1], self.partitionIdx[j+2]
            max1, min1 = max(trustedM3[index1+1:index2+1]), min(trustedM3[index1+1:index2+1])
            max2, min2 = max(trustedM3[index2+1:index3+1]), min(trustedM3[index2+1:index3+1])
            # print(str(trustedM1[j]) + ', ' + str(trustedM1[j+1]))
            if min1 >= trustedM1[j] and max1 < trustedM1[j + 1] <= min2 and max2 < trustedM1[j + 2] and max1 < min2:
                passFlag = True
            else:
                passFlag = False
                break
        print(passFlag)

    def OneLevelPartition(self, inStructureId, samples, sampleSize, p, outStructureId1):
        if self.N <= self.M:
            return self.N, 1
        hatN = math.ceil((1 + 2 * self.BETA) * self.N)
        M_prime = math.ceil(self.M / (1 + 2 * self.BETA))
        r = math.ceil(math.log(hatN / self.M) / math.log(p))
        p0 = math.ceil(hatN / (self.M * math.pow(p, r - 1)))
        trustedM1 = self.quantileCal(samples, 0, sampleSize, p0)
        boundary1 = math.ceil(self.N / M_prime)
        boundary2 = math.ceil(M_prime / self.B)
        dataBoundary = boundary2 * self.B
        # TODO: floor operation
        smallSectionSize = self.M // p0
        bucketSize0 = boundary1 * smallSectionSize
        # TODO: check initial list operations
        self.data[outStructureId1] = [self.DUMMY] * (boundary1 * smallSectionSize * p0)
        blocks_done = 0
        total_blocks = math.ceil(self.N / self.B)
        k = -1
        trustedM3 = [self.DUMMY] * (boundary2 * self.B)
        total_time = 0
        for i in range(boundary1):
            for j in range(boundary2):
                s = time.time()
                if not (total_blocks - 1 - blocks_done):
                    k = 0
                else:
                    k = random.randrange(0, total_blocks - blocks_done)

                # print("k: " + str(k))
                Msize1 = min(self.B, self.N - k * self.B)
                trustedM3[j*self.B:j*self.B + Msize1] = self.opOneLinearScanBlock(k * self.B, [], Msize1, inStructureId, 0)
                shuffleB = [self.DUMMY] * self.B
                Msize2 = min(self.B, self.N - (total_blocks - 1 - blocks_done) * self.B)
                shuffleB[0:Msize2] = self.opOneLinearScanBlock((total_blocks - 1 - blocks_done) * self.B, [], Msize2, inStructureId, 0)
                self.opOneLinearScanBlock(k * self.B, shuffleB, self.B, inStructureId, 1)
                blocks_done += 1
                if blocks_done == total_blocks:
                    break
                e = time.time()
                total_time += e - s
            blockNum = self.moveDummy(trustedM3, dataBoundary)
            # TODO: check new quick sort, especially border params
            self.quickSort(trustedM3, 0, blockNum-1, trustedM1, 1, p0)
            # TODO: check is we need remove duplicate indexes
            self.partitionIdx.sort()
            self.partitionIdx.insert(0, -1)
            # self.checkSection(trustedM1, trustedM3)
            # self.partitionIdx.append(blockNum-1)
            for j in range(p0):
                index1, index2 = self.partitionIdx[j]+1, self.partitionIdx[j+1]
                writeBackNum = index2 - index1 + 1
                if writeBackNum > smallSectionSize:
                    print("Overflow in small section M/p0: " + str(writeBackNum))
                self.opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, trustedM3[index1:index2+1],
                                          writeBackNum, outStructureId1, 1, smallSectionSize - writeBackNum)
            trustedM3 = [self.DUMMY] * (boundary2 * self.B)
            self.partitionIdx.clear()

        if bucketSize0 > self.M:
            print("Each section size is greater than M, adjust parameters: " + str(bucketSize0))
        print("Random total time: " + str(total_time))
        return bucketSize0, p0

    def TwoLevelPartition(self, inStructureId, pivots, p, outStructureId1, outStructureId2):
        """
        Finish two level partition
        pivots: List[]
        Return: bucket size & #pivots
        """

    def ObliviousTightSort(self, inStructureId, inSize, outStructureId1, outStructureId2):
        print("In ObliviousTightSort")
        start = time.time()
        if inSize <= self.M:
            trustedM = []
            trustedM = self.opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0)
            trustedM.sort()
            self.data[outStructureId1] = [self.DUMMY] * inSize
            self.opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1)
            return outStructureId1

        trustedM2 = []
        print("In SampleTight")
        start1 = time.time()
        realNum = self.Sample(inStructureId, trustedM2, self.is_tight)
        end1 = time.time()
        print("Till Sample IOcost: " + str(self.IOcost / self.N * self.B))
        print("In OneLevelPartition")
        start2 = time.time()
        sectionSize, sectionNum = self.OneLevelPartition(inStructureId, trustedM2, realNum, self.P, outStructureId1)
        end2 = time.time()
        print("Till Partition IOcost: " + str(self.IOcost / self.N * self.B))
        self.data[outStructureId2] = [self.DUMMY] * inSize
        trustedM = []
        j = 0
        print("In final")
        start3 = time.time()
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1)
            j += k
        end3 = time.time()
        end = time.time()
        print("Till Final IOcost: " + str(self.IOcost / self.N * self.B))
        print("Total=%.3f,Sample=%.3f,Partition=%.3f,Final=%.3f" % (end-start, end1-start1, end2-start2, end3-start3))
        return outStructureId2

    def ObliviousTightSort2(self, inStructureId, inSize, sampleId, outStructureId1, outStructureId2):
        """
        Implement two level oqsort
        """
        pivots = self.SampleRec(inStructureId, self.is_tight, sampleId)
        sectionSize, sectionNum = self.TwoLevelPartition(inStructureId, pivots, self.P, outStructureId1, outStructureId2)
        # outStructureId1 will be the final output
        trustedM = []
        j = 0
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(j, trustedM, k, outStructureId1, 1)
            j += k
        return outStructureId1

    def ObliviousLooseSort(self, inStructureId, inSize, outStructureId1, outStructureId2):
        print("In ObliviousLooseSort")
        start = time.time()
        if inSize <= self.M:
            trustedM = []
            trustedM = self.opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0)
            trustedM.sort()
            self.opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1)
            return outStructureId1, inSize

        trustedM2 = []
        print("In SampleLoose")
        start1 = time.time()
        realNum = self.Sample(inStructureId, trustedM2, self.is_tight)
        end1 = time.time()
        print("Till Sample IOcost: " + str(self.IOcost / self.N * self.B))
        print("In OneLevelPartition")
        start2 = time.time()
        sectionSize, sectionNum = self.OneLevelPartition(inStructureId, trustedM2, realNum, self.P, outStructureId1)
        end2 = time.time()
        print("Till Partition IOcost: " + str(self.IOcost / self.N * self.B))
        totalLevelSize = sectionSize * sectionNum
        self.data[outStructureId2] = [self.DUMMY] * totalLevelSize
        trustedM = []
        print("In final")
        start3 = time.time()
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1)
        end3 = time.time()
        end = time.time()
        print("Till Final IOcost: " + str(self.IOcost / self.N * self.B))
        print("Total=%.3f,Sample=%.3f,Partition=%.3f,Final=%.3f" % (
        end - start, end1 - start1, end2 - start2, end3 - start3))
        return outStructureId2, totalLevelSize

    def ObliviousLooseSort2(self, inStructureId, inSize, sampleId, outStructureId1, outStructureId2):
        """
        Implement two level oqsort
        """
        pivots = self.SampleRec(inStructureId, self.is_tight, sampleId)
        sectionSize, sectionNum = self.TwoLevelPartition(inStructureId, pivots, self.P, outStructureId1, outStructureId2)
        totalLevelSize = sectionSize * sectionNum
        self.data[outStructureId1] = [self.DUMMY] * totalLevelSize

        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 1)

        return outStructureId1, totalLevelSize

    def ObliviousLooseSortRec(self, sampleId, inSize, sortedSampleId):
        """
        called in SampleRec, sampleId inSize number having DUMMY
        sortedSampleId: store outputs after partition
        Inner sample func, use _alpha etc params
        Return: Selected Pivots: List
        """
        trustedM2 = []
        # TODO: ERROR sampleId numbers having DUMMY elements? How to fix
        realNum = self.Sample(sampleId, trustedM2, 0, 1)
        sectionSize, sectionNum = self.OneLevelPartition(sampleId, trustedM2, realNum, )


if __name__ == '__main__':
    N, M, B, is_tight = 5000000, 555556, 4, 0
    sortCase1 = OQSORT(N, M, B, 0, N)
    sortCase1.init(0, N)
    if N / M < 100:
        # is_tight flag
        sortCase1.onelevel(is_tight)
        print("Start running...")
        sortCase1.call()
        print("Finished.")
    else:
        # TODO: 2 level execution
        sortCase1.twolevel(is_tight)
        print("Start running...")
        sortCase1.call2()
        print("Finished.")


