from tracemalloc import start
from Crypto.Cipher import AES
import numpy as np
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
        # self.DUMMY = 0xffffffff
        self.DUMMY = '0xffffffff\n'
        # inStructureId, sampleId, sortedSampleId, outStructureId1, outStructureId2
        self.data = [[], [], [], [], []]
        # IO cost counting
        self.IOcost = 0

    def OcallReadBlock(self, index, blockSize, structureId):
        ret = self.data[structureId][index:index+blockSize]
        self.IOcost += math.ceil(blockSize/self.B)
        return ret

    def OcallWriteBlock(self, index, buffer, blockSize, structureId):
        self.data[structureId][index:index+blockSize] = buffer
        self.IOcost += math.ceil(blockSize/self.B)

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
        # random.shuffle(self.data[structureId])
    
    def readData(self, structureId):
        f = open("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/input2000")
        self.data[structureId] = f.readlines()
        f.close()

    def outResult(self, structureId, size):
        f = open("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/output2000", "w")
        self.data[structureId][0:size] = [x[:-1]+'\r\n' for x in self.data[structureId][0:size]]
        f.writelines(self.data[structureId][0:size])
        f.close()

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

class FFSEM:
    """
    Paper: Feistel Finite Set Encryption Mode
    """
    def __init__(self, key: bytes, max_num: int, rounds: int = 10):
        self.cipher = AES.new(key, AES.MODE_ECB)
        self._base = math.ceil(math.log2(max_num) / 2)
        self.max_num = 1 << 2*self._base
        if self.max_num != max_num:
            print("Permuted range is: " + str(self.max_num))
        self.rounds = rounds
    
    def prf(self, a: int) -> int:
        b = a.to_bytes(self.cipher.block_size, byteorder='big', signed=False)
        return int.from_bytes(self.cipher.encrypt(b), byteorder='big', signed=False)

    def encrypt(self, a: int) -> int:
        if a >= self.max_num:
            raise ValueError(f'Value must be less than {self.max_num}')

        l, r = divmod(a, 1 << self._base)
        for i in range(1, self.rounds+1):
            b = (r << self.cipher.block_size * 8 - self._base) + i
            e = self.prf(b)
            l, r = r, l ^ (e >> self.cipher.block_size * 8 - self._base)
        return (l << self._base) + r

    def decrypt(self, a: int) -> int:
        if a >= self.max_num:
            raise ValueError(f'Value must be less than {self.max_num}')
        
        l, r = divmod(a, 1 << self._base)
        for i in range(self.rounds, 0, -1):
            b = (l << self.cipher.block_size * 8 - self._base) + i
            e = self.prf(b)
            r, l = l, r ^ (e >> self.cipher.block_size * 8 - self._base)
        return (l << self._base) + r

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
        if is_tight:
            cost = 7 + 4 * beta
        else:
            cost = 6 + 6 * beta + alpha * self.B
        print("alpha=%f, beta=%f, p=%d, cost=%f" % (alpha, beta, p, cost))
        self.ALPHA, self.BETA, self.P, self.IdealCost = alpha, beta, p, cost
        self.sortId = not is_tight
        self.is_tight = is_tight
        return self.ALPHA, self.BETA, self.P, self.IdealCost

    # TODO: Two sets of alpha, beta, p
    def twolevel(self, is_tight, kappa=27.8):
        print("Calculating Parameters")
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
        self._alpha, self._beta, self._p, self._cost = self.onelevel(False, kappa + 1)
        # print("alpha1=%f, beta1=%f, p1=%d, cost1=%f" % (_alpha, _beta, _p, _cost))
        if is_tight:
            cost = 9 + 7 * alpha + 8 * beta
        else:
            cost = 8 + (6 + self.B) * alpha + 10 * beta
        self.ALPHA, self.BETA, self.P, self.IdealCost = alpha, beta, p, cost
        print("alpha=%f, beta=%f, p=%d, cost=%f" % (alpha, beta, p, cost))
        self.sortId = not is_tight
        self.is_tight = is_tight

    def call(self):
        """
        Method to Call OQSORT
        :return:
        """
        if not self.sortId:
            self.resId = self.ObliviousTightSort(self.structureId, self.paddedSize, self.structureId + 1,self.structureId + 2)
            # self.test(self.resId, self.paddedSize)
            self.outResult(self.resId, self.paddedSize)
        else:
            self.resId, self.resN = self.ObliviousLooseSort(self.structureId, self.paddedSize, self.structureId + 1, self.structureId + 2)
            self.testWithDummy(self.resId, self.resN)
            # self.outResult(self.resId, self.paddedSize)
        '''
        f = open("/homes/bchenba/OQSORT/out.txt", "w")
        for idx in range(self.resN//2, self.resN//2+1000000):
            f.write(str(self.data[self.resId][idx]) + str(' '))
            if not (idx % 10):
                f.write('\n')

        f.close()
        '''
    def call2(self):
        """
        Method to Call Two-Level OQSORT
        :return:
        """
        if not self.sortId:
            self.resId = self.ObliviousTightSort2(self.structureId, self.paddedSize, self.structureId + 1, self.structureId + 2, self.structureId + 3, self.structureId + 4)
            self.test(self.resId, self.paddedSize)
        else:
            self.resId, self.resN = self.ObliviousLooseSort2(self.structureId, self.paddedSize, self.structureId + 1, self.structureId + 2, self.structureId + 3, self.structureId + 4)
            self.testWithDummy(self.resId, self.resN)

        f = open("/Users/apple/Desktop/Lab/ALLSORT/ALLSORT/OQSORT/OQSORT/outputpy.txt", "w")
        for idx in range(self.resN):
            f.write(str(self.data[self.resId][idx]) + str(' '))
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

    def floydSampler(self, n, k):
        H = set()
        x = np.arange(n-k, n)
        for i in range(k):
            r = random.randint(0, n-k+1+i)
            if r in H:
                j = random.randint(0, i)
                x[i], x[j] = x[j], x[i]
                H.add(n-k+i)
            else:
                x[i] = r
                H.add(r)
        x.sort()
        return x
    # TODO: Read Size = M, How to allocate M
    def Sample(self, inStructureId, sampleSize, trustedM2, is_tight, is_rec=0):
        N_prime = sampleSize
        alpha = self.ALPHA if not is_rec else self._alpha
        n_prime = math.ceil(alpha * N_prime)
        boundary = math.ceil(N_prime / self.B)
        j = 0  # sampleIdx index
        trustedM1 = []
        sampleIdx = self.floydSampler(N_prime, n_prime)

        for i in range(boundary):
            if is_tight:
                trustedM1 = self.opOneLinearScanBlock(i * self.B, trustedM1, min(self.B, N_prime - i * self.B), inStructureId, 0)
                while j < n_prime and (sampleIdx[j] >= i * self.B) and (sampleIdx[j] < (i+1) * self.B):
                    trustedM2.append(trustedM1[sampleIdx[j] % self.B])
                    j += 1
            elif (not is_tight) and (sampleIdx[j] >= i * self.B) and (sampleIdx[j] < (i+1) * self.B):
                trustedM1 = self.opOneLinearScanBlock(i * self.B, trustedM1, min(self.B, N_prime - i * self.B), inStructureId, 0)
                while (sampleIdx[j] >= i * self.B) and (sampleIdx[j] < (i+1) * self.B):
                    trustedM2.append(trustedM1[sampleIdx[j] % self.B])
                    j += 1
                    if j >= n_prime:
                        break
                if j >= n_prime:
                    break

        # trustedM2.sort()
        trustedM2.sort(key=lambda x: x[0:10])
        # print(trustedM2)
        return n_prime

    '''TODO: Return all the pivots needed in two levels'''
    def SampleRec(self, inStructureId, sampleId, sortedSampleId, is_tight):
        """
        Finish sample tight recursive selection
        """
        N_prime = N
        n_prime = math.ceil(self.ALPHA * N_prime)
        boundary = math.ceil(self.N / self.M)
        realNum = 0
        readStart = 0
        trustedM1 = []
        m = 0
        self.data[sampleId] = [self.DUMMY] * n_prime

        for i in range(boundary):
            Msize = min(self.M, self.N - i * self.M)
            # print(type(N_prime), type(n_prime), type(Msize), type((int)(n_prime)))
            m = hypergeom.rvs(N_prime, (int)(n_prime), Msize, size=1)[0]
            if is_tight or (not is_tight and m > 0):
                trustedM1 = self.opOneLinearScanBlock(readStart, trustedM1, Msize, inStructureId, 0)
                readStart += Msize
                sams = random.sample(trustedM1, m)
                self.opOneLinearScanBlock(realNum, sams, m, sampleId, 1)
                realNum += m
                n_prime -= m
            if not (i % 10):
                print(i)
            N_prime -= Msize
        # TODO: contains DUMMY
        pivots = []
        if realNum > self.M:
            pivots = self.ObliviousLooseSortRec(sampleId, realNum, sortedSampleId)
        return pivots

    def quantileCal(self, samples, start, end, p):
        sampleSize = end - start
        for i in range(1, p):
            samples[i] = samples[i * sampleSize // p]
        samples[0] = chr(0) * 10
        samples[p] = chr(127) * 10
        return samples[0:p + 1]

    def partition(self, arr, low, high, pivot):
        """
        Return partition border index
        """
        i = low - 1
        '''
        for j in range(low, high+1):
            if pivot > arr[j]:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        '''
        # New Input
        for j in range(low, high+1):
            if pivot[0:10] > arr[j][0:10]:
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

    def OneLevelPartition(self, inStructureId, inSize, samples, sampleSize, p, outStructureId1):
        if inSize <= self.M:
            return self.N, 1
        # TODO: change self.N
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
        trustedM3 = [self.DUMMY] * (boundary2 * self.B)
        # Set up for pseudorandom permutation
        blocks_done = 0
        total_blocks = math.ceil(self.N / self.B)
        key = b'1Level Partition'
        codec = FFSEM(key, max_num=total_blocks)
        index_range = codec.max_num
        k = 0
        s = time.time()
        for i in range(boundary1):
            for j in range(boundary2):
                read_index = codec.encrypt(k)
                while read_index >= total_blocks:
                    k += 1
                    if k == index_range:
                        k = -1
                    read_index = codec.encrypt(k)  
                if k == -1: # Read finish
                    break
                Msize1 = min(self.B, self.N - read_index * self.B)
                trustedM3[j*self.B:j*self.B + Msize1] = self.opOneLinearScanBlock(read_index * self.B, [], Msize1, inStructureId, 0)
                k += 1
                if k == index_range: # Read finish
                    break
            blockNum = self.moveDummy(trustedM3, dataBoundary)
            # TODO: check new quick sort, especially border params
            self.quickSort(trustedM3, 0, blockNum-1, trustedM1, 1, p0)
            # TODO: check is we need remove duplicate indexes
            self.partitionIdx.sort()
            self.partitionIdx.insert(0, -1)
            # self.checkSection(trustedM1, trustedM3)
            for j in range(p0):
                index1, index2 = self.partitionIdx[j]+1, self.partitionIdx[j+1]
                writeBackNum = index2 - index1 + 1
                if writeBackNum > smallSectionSize:
                    print("Overflow in small section M/p0: " + str(writeBackNum))
                self.opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, trustedM3[index1:index2+1], writeBackNum, outStructureId1, 1, smallSectionSize - writeBackNum)
            trustedM3 = [self.DUMMY] * (boundary2 * self.B)
            self.partitionIdx.clear()
            if k == index_range: # Read finish
                break
        e = time.time()
        print("Time: " + str(e - s))
        if bucketSize0 > self.M:
            print("Each section size is greater than M, adjust parameters: " + str(bucketSize0))
        return bucketSize0, p0

    # FIXME: check
    def TwoLevelPartition(self, inStructureId, pivots, p, outStructureId1, outStructureId2):
        """
        Finish two level partition
        pivots: List[]
        Return: bucket size & #pivots
        """
        hatN = math.ceil((1 + 2 * self.BETA) * self.N)
        M_prime = math.ceil(self.M / (1 + 2 * self.BETA))
        p0 = p
        boundary1 = math.ceil(self.N / M_prime)
        boundary2 = math.ceil(M_prime / self.B)
        dataBoundary = boundary2 * self.B
        smallSectionSize = self.M // p0
        bucketSize0 = boundary1 * smallSectionSize
        self.data[outStructureId1] = [self.DUMMY] * (boundary1 * smallSectionSize * p0)
        blocks_done = 0
        total_blocks = math.ceil(self.N / self.B)
        trustedM3 = [self.DUMMY] * (boundary2 * self.B)
        # First Level
        for i in range(boundary1):
            for j in range(boundary2):
                if not (total_blocks - 1 - blocks_done):
                    k = 0
                else:
                    k = random.randrange(0, total_blocks - blocks_done)
                Msize1 = min(self.B, self.N - k * self.B)
                trustedM3[j * self.B:j * self.B + Msize1] = self.opOneLinearScanBlock(k * self.B, [], Msize1, inStructureId, 0)  # type: ignore
                shuffleB = [self.DUMMY] * self.B
                Msize2 = min(self.B, self.N - (total_blocks - 1 - blocks_done) * self.B)
                shuffleB[0:Msize2] = self.opOneLinearScanBlock((total_blocks - 1 - blocks_done) * self.B, [], Msize2, inStructureId, 0)  # type: ignore
                self.opOneLinearScanBlock(k * self.B, shuffleB, self.B, inStructureId, 1)
                blocks_done += 1
                if blocks_done == total_blocks:
                    break
            blockNum = self.moveDummy(trustedM3, dataBoundary)
            self.quickSort(trustedM3, 0, blockNum - 1, pivots[0], 1, p0)
            self.partitionIdx.sort()
            self.partitionIdx.insert(0, -1)
            for j in range(p0):
                index1, index2 = self.partitionIdx[j] + 1, self.partitionIdx[j + 1]
                writeBackNum = index2 - index1 + 1
                if writeBackNum > smallSectionSize:
                    print("Overflow in small section M/p0: " + str(writeBackNum))
                self.opOneLinearScanBlock(j * bucketSize0 + i * smallSectionSize, trustedM3[index1:index2 + 1], writeBackNum, outStructureId1, 1, smallSectionSize - writeBackNum)
            trustedM3 = [self.DUMMY] * (boundary2 * self.B)
            self.partitionIdx.clear()
        # Second Level
        p1 = p0 * p
        boundary3 = math.ceil(bucketSize0 / M)
        bucketSize1 = boundary3 * smallSectionSize
        self.data[outStructureId2] = [self.DUMMY] * (boundary3 * smallSectionSize * p0 * p)
        for j in range(0, p0):
            trustedM1 = pivots[1][j]
            for k in range(0, boundary3):
                Msize = min(self.M, bucketSize0 - k * self.M)
                readSize = Msize if Msize < (p+1) else Msize-(p+1)
                trustedM2 = self.opOneLinearScanBlock(k*self.M, [], readSize, outStructureId1, 0)
                k1 = self.moveDummy(trustedM2, readSize)
                trustedM2 = trustedM2[0:k1]
                readSize2 = 0 if Msize < (p+1) else p+1
                trustedM2_part = self.opOneLinearScanBlock(k*self.M+readSize, [], readSize2, outStructureId1, 0)
                k2 = self.moveDummy(trustedM2_part, readSize2)
                trustedM2.append(trustedM2_part[0:k2])
                self.quickSort(trustedM2, 0, k1 + k2 - 1, trustedM1, 1, p)
                self.partitionIdx.sort()
                self.partitionIdx.insert(0, -1)
                for ll in range(p):
                    index1, index2 = self.partitionIdx[j]+1, self.partitionIdx[j+1]
                    writeBackNum = index2 - index1 + 1
                    if writeBackNum > smallSectionSize:
                        print("Overflow in small section M/p0: " + str(writeBackNum))
                    self.opOneLinearScanBlock((j*p0+ll)*bucketSize1+k*smallSectionSize, trustedM2[index1:index2+1], writeBackNum, outStructureId2, 1, smallSectionSize-writeBackNum)
                trustedM2 = [self.DUMMY] * self.M
                self.partitionIdx.clear()

        if bucketSize1 > self.M:
            print("Each section size is greater than M, adjust parameters: " + str(bucketSize1))
        return bucketSize1, p1

    def ObliviousTightSort(self, inStructureId, inSize, outStructureId1, outStructureId2):
        print("In ObliviousTightSort")
        start = time.time()
        if inSize <= self.M:
            trustedM = []
            trustedM = self.opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0)
            # trustedM.sort()
            trustedM.sort(key=lambda x: x[0:10])
            self.data[outStructureId1] = [self.DUMMY] * inSize
            self.opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1)
            return outStructureId1

        trustedM2 = []
        print("In SampleTight")
        start1 = time.time()
        realNum = self.Sample(inStructureId, self.N, trustedM2, self.is_tight)
        end1 = time.time()
        print("Till Sample IOcost: " + str(self.IOcost / self.N * self.B))
        print("In OneLevelPartition")
        start2 = time.time()
        sectionSize, sectionNum = self.OneLevelPartition(inStructureId, self.N, trustedM2, realNum, self.P, outStructureId1)
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
            trustedM_part = sorted(trustedM[0:k], key=lambda x: x[0:10])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(j, trustedM, k, outStructureId2, 1)
            j += k
        end3 = time.time()
        end = time.time()
        print("Till Final IOcost: " + str(self.IOcost / self.N * self.B))
        print("Total=%.3f,Sample=%.3f,Partition=%.3f,Final=%.3f" % (end-start, end1-start1, end2-start2, end3-start3))
        return outStructureId2

    def ObliviousTightSort2(self, inStructureId, inSize, sampleId, sortedSampleId, outStructureId1, outStructureId2):
        """
        Implement two level oqsort
        """
        start1 = time.time()
        print("In ObliviousTightSort2 && In SampleRec")
        pivots = self.SampleRec(inStructureId, sampleId, sortedSampleId, 1)
        end1 = time.time()
        print("Till Sample IOcost: " + str(self.IOcost / self.N * self.B))
        print("In TwoLevelPartition")
        start2 = time.time()
        sectionSize, sectionNum = self.TwoLevelPartition(inStructureId, pivots, self.P, outStructureId1, outStructureId2)
        end2 = time.time()
        print("Till Partition IOcost: " + str(self.IOcost / self.N * self.B))
        trustedM = []
        j = 0
        print("In Final")
        start3 = time.time()
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k], key=lambda x: x[0:10])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(j, trustedM, k, outStructureId1, 1)
            j += k
        end3 = time.time()
        print("Till Final IOcost: " + str(self.IOcost / self.N * self.B))
        print("Total=%.3f,Sample=%.3f,Partition=%.3f,Final=%.3f" % (end3-start1, end1-start1, end2-start2, end3-start3))
        return outStructureId1

    def ObliviousLooseSort(self, inStructureId, inSize, outStructureId1, outStructureId2):
        print("In ObliviousLooseSort")
        start = time.time()
        if inSize <= self.M:
            trustedM = []
            trustedM = self.opOneLinearScanBlock(0, trustedM, inSize, inStructureId, 0)
            # trustedM.sort()
            trustedM.sort(key=lambda x: x[0:10]) 
            self.opOneLinearScanBlock(0, trustedM, inSize, outStructureId1, 1)
            return outStructureId1, inSize

        trustedM2 = []
        print("In SampleLoose")
        start1 = time.time()
        realNum = self.Sample(inStructureId, self.N, trustedM2, self.is_tight)
        end1 = time.time()
        print("Till Sample IOcost: " + str(self.IOcost / self.N * self.B))
        print("In OneLevelPartition")
        start2 = time.time()
        sectionSize, sectionNum = self.OneLevelPartition(inStructureId, self.N, trustedM2, realNum, self.P, outStructureId1)
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
            trustedM_part = sorted(trustedM[0:k], key=lambda x: x[0:10])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 1)
        end3 = time.time()
        end = time.time()
        print("Till Final IOcost: " + str(self.IOcost / self.N * self.B))
        print("Total=%.3f,Sample=%.3f,Partition=%.3f,Final=%.3f" % (
        end - start, end1 - start1, end2 - start2, end3 - start3))
        return outStructureId2, totalLevelSize

    def ObliviousLooseSort2(self, inStructureId, inSize, sampleId, sortedSampleId, outStructureId1, outStructureId2):
        """
        Implement two level oqsort
        """
        print("In ObliviousTightSort2 && In SampleRec")
        start1 = time.time()
        pivots = self.SampleRec(inStructureId, sampleId, sortedSampleId, 0)
        end1 = time.time()
        print("Till Sample IOcost: " + str(self.IOcost / self.N * self.B))
        print("In TwoLevelPartition")
        start2 = time.time()
        sectionSize, sectionNum = self.TwoLevelPartition(inStructureId, pivots, self.P, outStructureId1, outStructureId2)
        end2 = time.time()
        print("Till Partition IOcost: " + str(self.IOcost / self.N * self.B))
        totalLevelSize = sectionSize * sectionNum
        self.data[outStructureId1] = [self.DUMMY] * totalLevelSize
        trustedM = []
        print("In final")
        start3 = time.time()
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId2, 0)
            k = self.moveDummy(trustedM, sectionSize)
            trustedM_part = sorted(trustedM[0:k], key=lambda x: x[0:10])
            trustedM[0:k] = trustedM_part
            self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, outStructureId1, 1)

        end3 = time.time()
        print("Till Final IOcost: " + str(self.IOcost / self.N * self.B))
        print("Total=%.3f,Sample=%.3f,Partition=%.3f,Final=%.3f" % (end3 - start1, end1 - start1, end2 - start2, end3 - start3))
        return outStructureId1, totalLevelSize

    def ObliviousLooseSortRec(self, sampleId, sampleSize, sortedSampleId):
        """
        called in SampleRec, sampleId inSize number having DUMMY
        sortedSampleId: store outputs after partition
        Inner sample func, use _alpha etc params
        Return: Selected Pivots: List
        """
        # TODO: FIX different input size error
        print("In ObliviousLooseSortRec")
        trustedM2 = []
        print("In Sample's Sample")
        realNum = self.Sample(sampleId, sampleSize, trustedM2, 0, 1)
        print("In OneLevelPartition")
        sectionSize, sectionNum = self.OneLevelPartition(sampleId, sampleSize, trustedM2, realNum, self._p, sortedSampleId)
        j, k, total = 0, 0, 0
        outj, inj = 0, 0
        trustedM = []
        quantileIdx = [i * sampleSize // self.P for i in range(1, self.P)]
        size = math.ceil(sampleSize / self.P)
        quantileIdx2 = []
        print("Calculating Pivots")
        for i in range(self.P):
            # Actual Index in samples
            index = [i * size + j * size // self.P for j in range(1, self.P)]
            quantileIdx2.append(index)
        # TODO: Sorted Samples to selct pivots
        pivots1 = []
        pivots2 = []
        pivots2_part = []
        for i in range(sectionNum):
            trustedM = self.opOneLinearScanBlock(i * sectionSize, trustedM, sectionSize, sortedSampleId)
            k = self.moveDummy(trustedM, sectionSize)
            total += k
            # Cal Level1 pivots
            while j < self.P-1 and quantileIdx[j] < total:
                pivots1.append(trustedM[quantileIdx[j]-(total-k)])
                j += 1
            # Cal Level2 pivots
            while outj < self.P:
                while inj < self.P-1 and quantileIdx2[outj][inj] < total:
                    pivots2_part.append(trustedM[quantileIdx2[outj][inj] % sectionSize])
                    inj += 1
                    # this small section ends
                    if inj == self.P-1:
                        inj = 0
                        outj += 1
                        pivots2.append(pivots2_part)
                        pivots2_part = []
                        break
                if outj == self.P:
                    break
            if j >= self.P-1 and outj >= self.P:
                break
        # TODO: STring COmpare Error, need changing
        pivots1.insert(0, float('-inf'))
        pivots1.append(float('inf'))
        for i in range(self.P):
            pivots2[i].insert(0, float('-inf'))
            pivots2[i].append(float('inf')) 
        return [pivots1, pivots2]


if __name__ == '__main__':
    # M=32MB 4194304
    # N, M, B, is_tight = 200000000, 2000000, 3, 0
    N, M, B, is_tight = 2000000, 222222, 4, 1
    sortCase1 = OQSORT(N, M, B, 0, N)
    if N / M < 100:
        # is_tight flag
        sortCase1.onelevel(is_tight)
        print("Start running...")
        # sortCase1.init(0, N)
        sortCase1.readData(0)
        sortCase1.call()
        print("Finished.")
    else:
        # TODO: 2 level execution
        sortCase1.twolevel(is_tight)
        print("Start running...")
        sortCase1.init(0, N)
        sortCase1.call2()
        print("Finished.")


