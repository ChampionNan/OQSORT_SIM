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
        # data lists
        self.data = [[], [], []]
        # IO cost counting
        self.IOcost = 0
        self.sampleFlag, self.sampleCost = 0, 0
        self.partitionFlag, self.partitionCost = 0, 0
        self.finalFlag, self.finalCost = 0, 0

    def OcallReadBlock(self, index, blockSize, structureId):
        if not blockSize:
            return []
        ret = self.data[structureId][index:index + blockSize]
        self.IOcost += 1
        if self.sampleFlag:
            self.sampleCost += 1
        if self.partitionFlag:
            self.partitionCost += 1
        if self.finalFlag:
            self.finalCost += 1
        return ret

    def OcallWriteBlock(self, index, buffer, blockSize, structureId):
        if not blockSize:
            return []
        self.data[structureId][index:index + blockSize] = buffer
        self.IOcost += 1
        if self.sampleFlag:
            self.sampleCost += 1
        if self.partitionFlag:
            self.partitionCost += 1
        if self.finalFlag:
            self.finalCost += 1

    # TODO: check all this function calling
    def opOneLinearScanBlock(self, index, block, blockSize, structureId, write=0):
        boundary = (blockSize + self.B - 1) // self.B
        if not write:
            total = []
            for i in range(boundary):
                Msize = min(self.B, blockSize - i * self.B)
                part = self.OcallReadBlock(index + i * self.B, Msize, structureId)
                total.extend(part)
            return total
        else:
            for i in range(boundary):
                Msize = min(self.B, blockSize - i * self.B)
                self.OcallWriteBlock(index + i * self.B, block[i * self.B:i * self.B + Msize], Msize, structureId)

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
        self.resN = -1
        self.sortId = -1
        self.is_tight = -1
        # TODO: OQSORT params
        self.ALPHA, self.BETA, self.P, self.IdealCost = -1, -1, -1, -1

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
        start = time.time()
        if not self.sortId:
            self.resId = self.ObliviousTightSort(self.structureId, self.paddedSize, self.structureId + 1,
                                                 self.structureId + 2)
            self.test(self.resId, self.paddedSize)
        else:
            self.resId, self.resN = self.ObliviousLooseSort(self.structureId, self.paddedSize, self.structureId + 1,
                                                            self.structureId + 2)
            self.testWithDummy(self.resId, self.resN)
        end = time.time()
        print("Execution time: " + str(end - start))
        print("Total Cost, Sample Cost, Partition Cost, Final Cost: ")
        print(str(self.IOcost / self.N * self.B) + str(', ') +
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

    def Sample(self, inStructureId, trustedM2, is_tight):
        self.sampleFlag = 1
        N_prime = self.N
        n_prime = math.ceil(self.ALPHA * self.N)
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
                random.shuffle(trustedM1)
                trustedM2.extend(trustedM1[0:m])
                realNum += m
                n_prime -= m
            N_prime -= Msize

        trustedM2.sort()
        print(str(realNum) + ', ' + str(self.ALPHA * self.N))
        self.sampleFlag = 0
        return realNum

    def quantileCal(self, samples, start, end, p):
        sampleSize = end - start
        for i in range(1, p):
            samples[i] = samples[i * sampleSize // p]
        samples[0] = float('-inf')
        samples[p] = float('inf')
        return samples[0:p + 1]

    def BSFirstGE(self, array, size, target):
        left, right = 0, size - 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        if left >= size:
            return -1
        return left

    def BSFirstL(self, array, size, target):
        left, right = 0, size - 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if array[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        if right < 0:
            return -1
        return right

    def OneLevelPartition(self, inStructureId, samples, sampleSize, p, outStructureId1):
        self.partitionFlag = 1
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
        bucketNum = [[0 for i in range(boundary1)] for j in range(p0)]
        # TODO: check initial list operations
        self.data[outStructureId1] = [self.DUMMY] * (boundary1 * smallSectionSize * p0)

        blocks_done = 0
        total_blocks = math.ceil(self.N / self.B)
        k = -1
        trustedM3 = [self.DUMMY] * (boundary2 * self.B)
        for i in range(boundary1):
            for j in range(boundary2):
                if not (total_blocks - 1 - blocks_done):
                    k = 0
                else:
                    k = random.randint(0, total_blocks - 2 - blocks_done)
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
            blockNum = self.moveDummy(trustedM3, dataBoundary)
            trustedM3_part = sorted(trustedM3[0:blockNum])
            trustedM3[0:blockNum] = trustedM3_part

            for j in range(p0):
                pivot1, pivot2 = trustedM1[j], trustedM1[j + 1]
                index1 = self.BSFirstGE(trustedM3, blockNum, pivot1)
                index2 = self.BSFirstL(trustedM3, blockNum, pivot2)
                wNum = index2 - index1 + 1
                self.opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, trustedM3[index1:index2 + 1], wNum,
                                          outStructureId1, 1)
                bucketNum[j][i] += wNum
                if bucketNum[j][i] > smallSectionSize:
                    print("Overflow in small section M/p0: " + str(bucketNum[j][i]))
            trustedM3 = [self.DUMMY] * (boundary2 * self.B)

        for i in range(p0):
            for j in range(boundary1):
                self.padWithDummy(outStructureId1, j * bucketSize0 + i * smallSectionSize, bucketNum[j][i],
                                  smallSectionSize)

        if bucketSize0 > self.M:
            print("Each section size is greater than M, adjust parameters: " + str(bucketSize0))
        self.partitionFlag = 0

        return bucketSize0, p0

    def ObliviousTightSort(self, inStructureId, inSize, outStructureId1, outStructureId2):
        print("In ObliviousTightSort")
        if inSize <= self.M:
            trustedM = []
            trustedM = self.opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0)
            trustedM.sort()
            self.data[outStructureId1] = [self.DUMMY] * inSize
            self.opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1)
            return outStructureId1

        trustedM2 = []
        print("In SampleTight")
        realNum = self.Sample(inStructureId, trustedM2, self.is_tight)
        print("In OneLevelPartition")
        sectionSize, sectionNum = self.OneLevelPartition(inStructureId, trustedM2, realNum, self.P, outStructureId1)
        self.data[outStructureId2] = [self.DUMMY] * inSize
        trustedM = []
        j = 0
        self.finalFlag = 1
        print("In final")
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1)
            j += k

        self.finalFlag = 0
        return outStructureId2

    def ObliviousLooseSort(self, inStructureId, inSize, outStructureId1, outStructureId2):
        print("In ObliviousLooseSort")
        if inSize <= self.M:
            trustedM = []
            trustedM = self.opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0)
            trustedM.sort()
            self.opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1)
            return outStructureId1, inSize

        trustedM2 = []
        print("In SampleLoose")
        realNum = self.Sample(inStructureId, trustedM2, self.is_tight)
        print("In OneLevelPartition")
        sectionSize, sectionNum = self.OneLevelPartition(inStructureId, trustedM2, realNum, self.P, outStructureId1)
        totalLevelSize = sectionSize * sectionNum
        self.data[outStructureId2] = [self.DUMMY] * totalLevelSize
        trustedM = []
        self.finalFlag = 1
        print("In final")
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1)

        self.finalFlag = 0
        return outStructureId2, totalLevelSize


if __name__ == '__main__':
    N, M, B = 5000000, 555556, 4
    sortCase1 = OQSORT(N, M, B, 0, N)
    # is_tight flag
    sortCase1.onelevel(0)
    sortCase1.init(0, N)
    print("Start running...")
    sortCase1.call()
    print("Finished.")
