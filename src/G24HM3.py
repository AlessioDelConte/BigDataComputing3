import random
import sys

from pyspark.mllib.linalg import Vectors, DenseVector


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


def G24HM3(file_name, k, iter):
    P = readVectorsSeq(file_name)
    WP = [1.0] * len(P)
    P, C = kmeansPP(P, WP, k, iter)
    Lloyd(P, WP, partition(P, C), iter)


def partition(P: set, S: set):
    partitions = dict()
    for center in S:
        partitions[center] = {center}
    for p in P - S:
        minimum = sys.maxsize
        index = None
        for center in partitions.keys():
            if p.dst(center) < minimum:
                minimum = p.dst(center)
                index = center
        partitions[index].add(p)
    return partitions


def centroid(partitions: dict):
    partitions_with_new_centers = dict()
    for center, cluster in partitions.items():
        new_centroid = Point(DenseVector([0.0, 0.0]), 1)
        for point in cluster:
            new_centroid.coordinates += point.coordinates
        new_centroid.coordinates *= 1 / len(cluster)
        partitions_with_new_centers[new_centroid] = cluster
    return partitions_with_new_centers


def Lloyd(P, WP, partitions, iter):
    i = 1
    stop = False
    prev_phi = sys.maxsize
    while i <= iter and not stop:
        partitions_with_new_centroids = centroid(partition(P, {center for center in partitions.keys()}))
        phi = kmeansObj(P, partitions_with_new_centroids.keys())
        if phi < prev_phi:
            prev_phi = phi
            partitions = partitions_with_new_centroids
        else:
            stop = True
        i += 1
    return partitions


def kmeansPP(P, WP, k, iter):
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

        pool = P - S

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
    return P, S


def kmeansObj(P, C):
    accumulator = 0
    for point in P:
        accumulator += min([point.dst(center) for center in C])
    return accumulator / len(P)


def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list


G24HM3(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
