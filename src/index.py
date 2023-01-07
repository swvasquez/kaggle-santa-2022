"""
Use this file to generate an index for configurations. This, however, is 
primarily useful for a smaller number of links, as the number of configurations
is extremely large in the general case.
"""

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils import start_cfg


def square(n, start):
    order = {
                (0, -1): (0, 1),
                (1, -1): None,
                (1, 0): (1, 1),
                (1, 1): None,
                (0, 1): (0, -1),
                (-1, 1): None,
                (-1, 0): (1, -1),
                (-1, -1): None
    }
    points = defaultdict(list)
    for x in itertools.product(range(-n, n + 1), repeat=2):
        if max(abs(x[0]), abs(x[1])) == n:
            normed = (int(x[0] / n), int(x[1] / n))
            points[normed].append(x)
    for k in points:
        if order[k] is not None:
            idx, direction = order[k]
            points[k].sort(key=lambda x:x[idx], reverse=(direction == -1))
    
    tail = []
    begin = False
    for o in order:
        for pt in points[o]:
            if not begin:
                if pt == start:
                    begin = True
                else:
                    tail.append(pt)
            if begin:
                yield pt

    for pt in tail:
        yield pt


def _index(n_links):
    start = start_cfg(n_links)
    start = [(start[i], start[i + 1]) for i in range(0, len(start) - 1, 2)]
    squares = (square(max(abs(x[0]), abs(x[1])), x) for x in start)
    for idx, sq in enumerate(itertools.product(*squares)):
        yield idx, sum((list(s) for s in sq), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n-links", type=int)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()

    n_links = max(args.n_links, 2)
    output_dir = Path(args.output_dir) / f"{n_links}-links"
    output_dir.mkdir(exist_ok=True, parents=True)
   
    output_path = output_dir / f"{n_links}-index.npy"

    cfgs = []
    for idx, cfg in _index(n_links):
        cfgs.append(cfg)
        print(idx, cfg)
    
    cfgs = np.array(cfgs, dtype=np.short)
    np.save(output_path, cfgs)