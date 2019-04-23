import random
import sys

from pyspark.mllib.linalg import Vectors


def G24HM3(file_name, k, iter):
    P = readVectorsSeq(file_name)
    WP = [1.0] * len(P)
    kmeansPP(P, WP, int(k), int(iter))


def kmeansPP(P, WP, k, iter):
    def Lloyd():
        pass

    c0 = P.pop(random.randrange(0, len(P)))
    S = [[c0, 0, 0]]
    for i in range(0, len(P)):
        P[i] = [P[i], 0, 0]

    for i in range(1, k):
        for point in P:
            point[1] = min([point[0].squared_distance(center[0]) for center in S])  # K^2 * |P| ??
        pool = set(tuple(i) for i in P) - set(tuple(i) for i in S)
        total = 0

        j = 0
        for pair in pool:
            total += WP[i] * pair[1]
            j += 1

        j = 0
        for pair in pool:
            P[j][2] = WP[j] * pair[1] / total
            j += 1

        acc = j = 0
        find = False
        x = random.random()
        while not find and j < len(P):
            acc += P[j][2]
            if acc >= x:
                find = True
            j += 1
        S.append(P[j - 1])
    return S


def kmeansObj(P, C):
    pass


def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list


G24HM3(sys.argv[1], sys.argv[2], sys.argv[3])
