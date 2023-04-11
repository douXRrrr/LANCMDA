'''
对单条序列进行切分
返回单条序列的Kmer
'''


def MyKmer(MySequence, k=3):         # k是几mer，s是取前几个奇异值
    # 建立空KmerRow
    KmerRow = []
    counter = 0
    while counter < 4**k:
        KmerRow.append(0)
        counter = counter + 1
    # 向矩阵填1
    counter = 0
    while counter < len(MySequence) - k + 1:
        sequence = MySequence[counter:counter + k]
        # print(sequence)
        num = 0
        n = 0
        while n < k:
            if sequence[k - n - 1] == 'A':
                num = num + 0 * 4**n
            if sequence[k - n - 1] == 'C':
                num = num + 1 * 4**n
            if sequence[k - n - 1] == 'G':
                num = num + 2 * 4**n
            if sequence[k - n - 1] == 'U':
                num = num + 3 * 4**n
            n = n + 1
        # print(num)
        KmerRow[num] = KmerRow[num] + 1
        counter = counter + 1
    counter = 0
    while counter < len(KmerRow):
        KmerRow[counter] = KmerRow[counter] / (len(MySequence) - k + 1)
        counter = counter + 1
    return KmerRow