import random
import sys
# import numpy as np

from pyspark.mllib.linalg import Vectors


def G24HM3(file_name, k, iter):
    P = readVectorsSeq(file_name)
    WP = [1.0] * len(P)
    kmeansPP(P, WP, int(k), int(iter))


def kmeansPP(P, WP, k, iter):

    def Lloyd():
        pass

    c0 = random.choice(P)

    class Point:
        def __init__(self, coords):
            self.coordinates = coords
            self.dst_from_closest_center = sys.maxsize
            self.closest_center = None
            self.extraction_probability = 0

        def dst(self, other):
            assert isinstance(other, Point)
            return self.coordinates.squared_distance(other.coordinates)  # note we are using distances, not squared distances!

        # def __lt__(self, other):
        #    return id(self.coordinates) < id(other.coordinates)  # interessante per np.setdiff1d. da rimuovere.

    starting_center = Point(c0)
    S = {starting_center}  # si potrebbe passare tutto a set? discuterne.

    pointsP = []
    for coord in P:
        pointsP.append(Point(coord))

    P = set(pointsP)

    last_center_found = starting_center

    # from itertools import filterfalse
    # from collections import Counter

    for i in range(1, k):
        for point in P:
            dst_between_point_and_last_center = point.dst(last_center_found)
            if dst_between_point_and_last_center < point.dst_from_closest_center:
                point.dst_from_closest_center = dst_between_point_and_last_center
                point.closest_center = last_center_found

        pool = P - S  # <1s
        # pool = list(set(P) - set(S))  # 1s
        # stranamente list( ) prende pochissimo tempo. meglio.
        # direi di tenere questa implementazione visto che è la più veloce.

        # pool = list(filterfalse(set(S).__contains__, P))  # 1s
        # pool = list(filterfalse(set(S).__contains__, P))  # 1
        # pool = np.array(list(set(P) - set(S)))  # 1.17s
        # pool = np.array([x for x in P if x not in S]) 2.29s
        # pool = np.setdiff1d(P, S) #  -- 2.28s (K=20)
        #  https://gist.github.com/denfromufa/2821ff59b02e9482be15d27f2bbd4451  << ma non e' vero niente, wtf

        # vecchia implementazione: 15s

        total = 0

        for point in pool:
            total += WP[i] * point.dst_from_closest_center

        j = 0
        for point in pool:
            point.extraction_probability = WP[j] * point.dst_from_closest_center / total
            j += 1

        accumulator = 0
        found = False
        extracted_number = random.random()

        extracted_point = None
        for point in pool:
            accumulator += point.extraction_probability
            if accumulator >= extracted_number:
                found = True
                extracted_point = point
                break

        assert extracted_point is not None
        S.add(extracted_point)
    return S


def kmeansObj(P, C):
    pass


def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list

import time

start = time.time()
G24HM3(sys.argv[1], sys.argv[2], sys.argv[3])
print("Elapsed time: ", time.time() - start)

# tempo di esecuzione con K=1000: 37.8s
# su Intel Core i5-6600K @ 3.5GHz.
