from fastdtw import fastdtw

def find_closest_pair(cluster):
    """Find the most similar pair of signals in a given cluster."""
    if len(cluster) < 2:
        return (None, None), 0.0

    min_dist = float('inf')
    pair = (None, None)

    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dist, _ = fastdtw(cluster[i], cluster[j])
            if dist < min_dist:
                min_dist = dist
                pair = (i, j)

    return pair, min_dist
