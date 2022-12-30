import random
import time
from collections import Counter

from copy import deepcopy


def search(start, neighborhoods, costs, coords, cfg_path, px_path, min_visits,
           min_loop_size, max_steps, max_degree, cost, threshold, start_time, 
           timeout, seed):
    
    nbhd = neighborhoods[start]
    indices = list(range(int(nbhd.shape[0])))
    random.shuffle(indices)

    n_costs = costs[start]
    
    degree = 0
    if min_loop_size > 0:
        tail = px_path[(-1 * min_loop_size):]
        if len(set(tail)) == len(tail):
            degree = min(max_steps, max_degree)
    
    for idx in indices:
        nbr = nbhd[idx]
        point = coords[nbr]
        p_cost = cost + n_costs[idx]
        runtime = time.time() - start_time 
        if (threshold is None or p_cost < threshold) and runtime < timeout:
            visited = Counter(px_path)
            step = len(cfg_path)
            if visited[tuple(point)] <= degree:
                if step <= max_steps:
                    _cfg_path = deepcopy(cfg_path)
                    _px_path = deepcopy(px_path)
                    _cfg_path.append(int(nbr))
                    _px_path.append(tuple(point))
                    visited[tuple(point)] += 1
                if step < max_steps:
                    if len(visited) == min_visits:
                        yield _cfg_path, visited, p_cost
                    else:
                        _search = search(
                            start=nbr,
                            neighborhoods=neighborhoods,
                            costs=costs,
                            coords=coords,
                            cfg_path=_cfg_path,
                            px_path=_px_path,
                            min_visits=min_visits,
                            min_loop_size=min_loop_size,
                            max_steps=max_steps,
                            max_degree=max_degree,
                            cost=p_cost, 
                            threshold=threshold,
                            start_time=start_time, 
                            timeout=timeout, 
                            seed=seed

                        )
                        yield from _search
                elif step == max_steps:
                    yield _cfg_path, visited, p_cost
