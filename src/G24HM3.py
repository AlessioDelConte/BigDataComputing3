import random
import sys
from pprint import pprint

from pyspark.mllib.linalg import Vectors


def G24HM3(file_name, k, iter):
    P = readVectorsSeq(file_name)
    WP = [1.0] * len(P)
    kmeansPP(P, WP, int(k), int(iter))


def kmeansPP(P, WP, k, iter):
    class Point:
        def __init__(self, vector, dist, prob):
            self.vector = vector
            self.dist = dist
            self.prob = prob

        def __repr__(self):
            return str(self.vector)

    def Lloyd():
        pass

    # first random point chosen from P with uniform prob.
    c0 = Point(P.pop(random.randrange(0, len(P))), 0, 0)  # P[i] = point, dist_to_closest_center, p_to_be_extracted

    S = [c0]

    P_ = []
    for i in range(0, len(P)):
        point = Point(P[i], 0, 0)
        P_.append(point)

    for _ in range(1, k):
        for point in P_:
            point.dist = min([point.vector.squared_distance(center.vector) for center in S])  # K^2 * |P| ??

        pool = [point for point in P_ if point not in S]

        # calcolo denominatore formula prof (per la pool, che ha solo i non-centri, ovvero P-S).
        total = 0
        for i, point in enumerate(pool):
            total += WP[i] * point.dist

        # calcolo la probabilità per ogni punto di essere estratta.
        for i, point in enumerate(pool):
            point.prob = WP[i] * point.dist / total

        acc = 0
        x = random.random()
        for i, point in enumerate(pool):
            acc += point.prob
            if acc >= x:
                S.append(point)
                break
    print(S)
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
