import random
import sys
import time

from pyspark.mllib.linalg import Vectors


def G24HM3(file_name, k, iter):
    P = readVectorsSeq(file_name)
    WP = [1.0] * len(P)
    start = time.time()
    C = kmeansPP(P, WP, int(k), int(iter))
    print("Elapsed time: ", time.time() - start)
    start = time.time()
    print(kmeansObj(P, C))
    print("Elapsed time: ", time.time() - start)



class Point:
    def __init__(self, coords, weight=1):
        self.coordinates = coords
        self.dst_from_closest_center = sys.maxsize
        self.closest_center = None
        self.extraction_probability = 0
        self.weight = weight

    def dst(self, other):
        assert isinstance(other, Point)
        return self.coordinates.squared_distance(other.coordinates)
        # note we are using distances, not squared distances!


def kmeansPP(P, WP, k, iter):
    def Lloyd():
        pass

    # the first center is defined with uniform probability from P
    random_index = random.randrange(0, len(P))

    c0 = P[random_index]
    w0 = WP[random_index]
    starting_center = Point(c0, w0)
    S = {starting_center}

    pointsP = set()
    for coord, weight in zip(P, WP):
        pointsP.add(Point(coord, weight))

    P = pointsP  # using sets for more performance

    last_center_found = starting_center

    for i in range(1, k):
        for point in P:
            dst_between_point_and_last_center = point.dst(last_center_found)
            # updating minimum if we find a closer center
            if dst_between_point_and_last_center < point.dst_from_closest_center:
                point.dst_from_closest_center = dst_between_point_and_last_center
                point.closest_center = last_center_found

        pool = P - S  # <1s

        # calculate the denominator of the expression used to extract the probability
        total = sum([point.weight * point.dst_from_closest_center for point in pool])

        # assign the correct probability according to the formula
        for point in pool:
            point.extraction_probability = point.weight * point.dst_from_closest_center / total

        assert round(sum([point.extraction_probability for point in pool]), 3) == 1.0

        # random weighted extraction implementation
        accumulator = 0
        extracted_number = random.random()

        extracted_point = None
        for point in pool:
            accumulator += point.extraction_probability
            if accumulator >= extracted_number:
                extracted_point = point
                break

        assert extracted_point is not None

        last_center_found = extracted_point

        # this is the new center
        S.add(extracted_point)

    # Keep the distances updated with the last center selected
    for point in P:
        dst_between_point_and_last_center = point.dst(last_center_found)
        # updating minimum if we find a closer center
        if dst_between_point_and_last_center < point.dst_from_closest_center:
            point.dst_from_closest_center = dst_between_point_and_last_center
            point.closest_center = last_center_found

    return S


def kmeansObj(P, C):
    accumulator = 0
    for point in P:
        accumulator += min([point.squared_distance(center.coordinates) for center in C])
    return accumulator / len(P)


def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list


G24HM3(sys.argv[1], sys.argv[2], sys.argv[3])

# tempo di esecuzione con K=1000: 37.8s
# su Intel Core i5-6600K @ 3.5GHz.
