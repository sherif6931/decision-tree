import heapq
import numpy as np
from core.map import Map, Hex, Center
from itertools import count

def astar(p: Hex, Q: Hex, map: Map, h, visible):
    open_set = []
    counter = count()

    heapq.heappush(open_set, (0, next(counter), p))

    last = {}
    g_score = {p: 0}
    while open_set:
        i, j, current = heapq.heappop(open_set)

        if current == Q:
            path = [current]

            while current in last:
                current = last[current]

                path.append(current)

            return path[::-1]

        for nb in map.neighbor_hex(current):
            if nb is None:
                continue

            if visible is not None and nb not in visible:
                continue

            tentative_g = g_score[current] + map.costs[nb]
            if nb not in g_score or tentative_g < g_score[nb]:
                last[nb] = current
                g_score[nb] = tentative_g

                f_score = tentative_g + h(nb, Q)

                heapq.heappush(open_set, (f_score, next(counter), nb))

    return None

def rationalBias(p, map: Map, degree_of_bias=10):
    degree_of_bias = np.remainder(degree_of_bias,10)

    neighbors = [nb for nb in map.neighbor_hex(p) if nb is not None]
    T = [map.costs[nb] for nb in neighbors]

    t_max = max(T)
    S = [pow(t_max - t + 1, degree_of_bias) for t in T]

    St = [t / sum(S) for t in S]

    return np.random.choice(len(neighbors), p=St)