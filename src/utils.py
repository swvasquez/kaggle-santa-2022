import itertools
import random
from collections import defaultdict

import numpy as np
import matplotlib.collections as mc
import matplotlib.pyplot as plt
from PIL import Image

############################### Search Utilities ##############################

def links(n_links):
    lengths = []
    for l in reversed(range(n_links - 1)):
        lengths.append(2 ** l)
    lengths.append(1)
    
    return tuple(lengths)


def start_cfg(n_links):
    cfg = [0] * (2 * n_links)
    for idx, length in enumerate(links(n_links)):
        if idx == 0:
            cfg[idx] = length
        else:
            cfg[2*idx] = -length

    return tuple(cfg)


def distances(nbhd, cfg):
    return np.linalg.norm(nbhd - cfg, axis=1)


def random_cfg(n_links):
    lengths = links(n_links)
    out = []
    for l in lengths:
        link = [0,0]
        max_pos = random.randint(0,1)
        link[max_pos] = random.choice([-l, l])
        link[(max_pos + 1) % 2] = random.randint(-l, l)
        out += link

    return np.array(out, dtype=np.byte)


def locations(cfg, origin=np.array((0,0))):
    delta = np.sum(cfg.reshape(-1,2), dtype=np.int32, axis=0)
    return origin + delta
   
    
def cfgs_map(n_links, output_path):
    lengths = links(n_links)
    boundaries = []
    for l in lengths:
        boundary = []
        box =  itertools.product(range(-l, l + 1), range(-l, l + 1))
        for coord in box:
            if max(abs(coord[0]), abs(coord[1])) == l:
                boundary.append([coord[0], coord[1]])
        boundaries.append(boundary)
        
    n_cfgs = int(np.prod([8*n for n in lengths]))
    index = np.zeros((n_cfgs, 2 * n_links), dtype=np.int32)

    for idx, cfg in enumerate(itertools.product(*boundaries)):
        cfg = tuple(sum(cfg,[]))
        index[idx] = np.array(cfg)
    
    np.save(output_path, index)
      
    return index


def coords_map(cfgs, output_path, origin=np.array((0,0))):
    index = np.zeros((cfgs.shape[0], 2), dtype=np.int32)
    for idx, cfg in enumerate(cfgs):
        index[idx] = locations(cfg, origin)
    np.save(output_path, index)
    
    return index


def rotation_matrix(n_links):
    matrix = []
    cache = set()
    directions = itertools.product({-1,0,1}, repeat=n_links)
    for shifts in (list(d) for d in directions):
        if shifts != [0] * n_links:
            transforms = [[], []]
            for shift in shifts:
                transforms[0].append((shift, 0))
                transforms[1].append((0, shift))
            for choices in itertools.product({0,1}, repeat=n_links):
                transform = []
                for idx, choice in enumerate(choices):
                    transform.extend(transforms[choice][idx])
                if tuple(transform) not in cache:
                    matrix.append(transform)
                    cache.add(tuple(transform))                  
    
    matrix = np.array(matrix, dtype=np.int32)         
    
    return matrix


def is_valid(cfg, origin=np.array((0, 0))):
    pos, valid = in_bounds(cfg, origin)
    if valid:
        lengths = np.array(links(cfg.shape[0] // 2))
        maximums = np.abs(cfg.reshape(-1,2)).max(axis=1)
        valid &= np.all(np.equal(maximums, lengths))
    
    return pos, valid


def in_bounds(cfg, origin=np.array((0, 0))):
    pos = locations(cfg, origin)
    n_links = cfg.shape[0] // 2
    img_size = np.array([2 * sum(links(n_links)) + 1] * 2)

    return pos, bool(np.all(np.less_equal(np.abs(pos), img_size)))


def rotations_map(cfgs, output_path):
    n_links = cfgs.shape[1] // 2
    rotations = rotation_matrix(n_links)
    
    n_cfgs = cfgs.shape[0]
    r_map = np.zeros((n_cfgs, 3 ** n_links - 1, 2* n_links), dtype=np.int32)
    
    select = defaultdict(list)

    max_nbrs = 0
    for c_idx, cfg in enumerate(cfgs):
        for r_idx, nbr in enumerate(rotations + cfg):
            _, valid = is_valid(nbr)
            if valid: 
                select[c_idx].append(r_idx)
        max_nbrs = max(max_nbrs, len(select[c_idx]))
        
    assert max_nbrs == (3 ** n_links) - 1
    
    for c_idx, r_indices in select.items():
        r_map[c_idx] = rotations[r_indices]
        
    np.save(output_path, r_map)
        
    return r_map
    
    
def neighborhoods_map(cfgs, rotations, output_path):
    index = {tuple(cfg): idx for idx, cfg in enumerate(cfgs)}
    n_links = cfgs.shape[1] // 2
    n_cfgs = cfgs.shape[0]
    n_map = np.zeros((n_cfgs, 3 ** n_links - 1), dtype=np.int32)
    
    for c_idx, cfg in enumerate(cfgs):
        nbrs = rotations[c_idx] + cfg
        for n_idx, nbr in enumerate(nbrs):
            n_map[c_idx][n_idx] = index[tuple(nbr)]
        
    np.save(output_path, n_map)
    
    return n_map


def color_costs_map(img_path):
    img = np.asarray(Image.open(img_path)) / 255
    shape = img.shape[:-1]
    c_map = np.zeros(shape * 2)
    points = list(itertools.product(range(shape[0]), range(shape[1])))
    for pair in itertools.product(points, points):
        idx = sum([list(p) for p in pair], [])
        diff = img[tuple(idx[:2])] - img[tuple(idx[2:])]
        c_map[idx] = 3 * np.abs(diff).sum()
    
    return c_map


def cfg_costs_map(rotations):
    c_map = np.zeros(rotations.shape[:2], dtype=np.float32)
    for c_idx, rots in enumerate(rotations):
        c_map[c_idx] = np.sqrt(np.sum(np.abs(rots), axis=1, dtype=np.float32))
    
    return c_map
 

def pixels(coords, img_size):
    x = coords[0] + (img_size[1] // 2)
    y = (img_size[0] // 2) - coords[1]
    
    return np.array((x,y), dtype=np.int32)


def costs_map(img_path, cfgs, rotations, coords, output_path):
    color_costs = color_costs_map(img_path)
    cfg_costs = cfg_costs_map(rotations)
    
    n_links = cfgs.shape[1] // 2
    img_size = np.array([2 * sum(links(n_links)) + 1] * 2, dtype=np.int32)
    
    c_map = np.copy(cfg_costs)
    for c_idx, rots in enumerate(rotations):
        nbhd = rotations[c_idx] + cfgs[c_idx]
        c_coords = coords[c_idx]
        for n_idx, nbr in enumerate(nbhd):
            dest = list(pixels(locations(nbr), img_size)) 
            src = list(pixels(c_coords, img_size))
            color_cost = color_costs[tuple(src + dest)]
            c_map[c_idx][n_idx] += color_cost
    
    np.save(output_path, c_map)
    
    return c_map

    
def load_cfgs(cfgs_path):
    return np.load(cfgs_path)


def load_coords(coords_path):
    return np.load(coords_path)


def load_rotations(rotations_path):
    return np.load(rotations_path)


def load_neighborhoods(neighborhoods_path):
    return np.load(neighborhoods_path)


def load_costs(costs_path):
     return np.load(costs_path)


def load_data(n_links, src_path, output_dir, force=False):
    output_dir = output_dir / f"{n_links}-links"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cfgs_path = output_dir / f"{n_links}-configurations.npy"
    coords_path = output_dir / f"{n_links}-coordinates.npy"
    rotations_path = output_dir / f"{n_links}-rotations.npy"
    neighborhoods_path = output_dir/ f"{n_links}-neighborhoods.npy"
    costs_path = output_dir/ f"{n_links}-costs.npy"
    img_path = output_dir / f"{n_links}-{src_path.stem}{src_path.suffix}"
    
    img = resize(src_path, img_path, n_links)

    if not cfgs_path.is_file() or force:
        cfgs = cfgs_map(n_links=n_links, output_path=cfgs_path)
    else:
        cfgs = load_cfgs(cfgs_path)
    print(f"Configurations: {cfgs.shape}")

    if not coords_path.is_file() or force:
        coords = coords_map(cfgs, coords_path)
    else:
        coords = load_coords(coords_path)
    print(f"Coordinates: {coords.shape}")

    if not rotations_path.is_file() or force:
        rotations = rotations_map(cfgs, rotations_path)
    else:
        rotations = load_rotations(rotations_path)
    print(f"Rotations: {rotations.shape}")

    if not neighborhoods_path.is_file() or force:
        neighborhoods = neighborhoods_map(cfgs, rotations, neighborhoods_path)
    else:
        neighborhoods = load_neighborhoods(neighborhoods_path)
    print(f"Neighborhoods: {neighborhoods.shape}")

    if not costs_path.is_file() or force:
        costs = costs_map(img_path, cfgs, rotations, coords, costs_path)
    else:
        costs = load_costs(costs_path)
    print(f"Costs: {costs.shape}")

    start = start_cfg(n_links)
    start_idx = None
    
    for cfg_idx, cfg in enumerate(cfgs):
        if np.all(np.equal(np.array(start, dtype=np.int32), cfg)):
            start_idx = cfg_idx

    assert start_idx is not None
    
    return {
        "configurations": cfgs,
        "coordinates": coords,
        "costs": costs,
        "neighborhoods": neighborhoods,
        "rotations": rotations,
        "start": start_idx,
        "image": img
    }   

################################ Visualization ################################

def plot(cfg_path, coords, img, out_path, img_path=None):
    if img_path is not None:
        img = np.asarray(Image.open(img_path)) / 255
    lines = []
    for idx, cfg in enumerate(cfg_path):
        coord = coords[cfg]
        if idx > 0:
            lines.append([prev, coord])
        prev = coord
    colors = []
    for l in lines:
        dist = np.abs(l[0] - l[1]).max()
        if dist <= 2:
            colors.append('b')
        else:
            colors.append('r')
    lc = mc.LineCollection(lines, colors=colors)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.add_collection(lc)

    radius = img.shape[0] // 2
    ax.matshow(
        img * 0.8 + 0.2, 
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5)
    )
    ax.grid(None)

    ax.autoscale()
    plt.savefig(out_path)
    plt.close(fig)

    
def resize(img_path, output_path, n_links):
    img_size = np.array([2 * sum(links(n_links)) + 1] * 2)
    img = Image.open(img_path)
    img = img.resize(tuple(img_size))
    img.save(output_path)
    
    return np.asarray(img) / 255
