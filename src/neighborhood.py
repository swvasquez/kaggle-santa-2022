import itertools
from pathlib import Path

import numpy as np

from utils import links, start_cfg

def tuple_encoding():
    index = np.zeros((3,3), dtype=np.ubyte)
    inverse = np.zeros((9,2), dtype=np.short)
    deltas = list(itertools.product([-1,0,1], repeat=2))
    deltas.sort(key=lambda x: (abs(x[0]), abs(x[1])), reverse=True)
    for idx, delta in enumerate(deltas):
        index[delta] = idx
        inverse[idx] = delta

    return index, inverse


def rotations_index(n_links):
    r_decoder = []
    r_encoder = np.zeros([3] * (n_links * 2), dtype=np.uintc)
    link_rots = [(0,0), (-1,0), (0, -1), (1, 0), (0, 1)]
    for idx, x in enumerate(itertools.product(link_rots, repeat=n_links)):
        rotation = tuple(sum((list(_d) for _d in x), []))
        r_encoder[rotation] = idx
        r_decoder.append(rotation)
    r_decoder = np.array(r_decoder, dtype=np.byte)

    return r_encoder, r_decoder


def permutations_index(n_links):
    idx = []
    for perm in itertools.permutations(list(range(n_links))):
        p1 = 2 * np.array(perm, dtype=np.ubyte)
        p2 = p1 + 1
        idx.append(np.ravel(np.column_stack((p1, p2))))
       
    return np.array(idx, dtype=np.ubyte)
   

def ascending(n, lower, upper):
    if n == 1:
        for x in range(lower, upper):
            yield [x]
    else:
        for x in range(lower, upper):
            for y in ascending(n - 1, x , upper):
                yield [x] + y


def gen_data(n_links, output_dir, force=False):

    encoder_path = output_dir / f"{n_links}-encoder.npy"
    moves_path = output_dir / f"{n_links}-moves.npy"
    permutations_path = output_dir / f"{n_links}-permutations.npy"
    
    t_encoder, t_decoder = tuple_encoding()
    
    _link_moves = {
            (0, 1): [(-1, 0), (0, 0), (1, 0)],
            (1, 0): [(0, -1), (0, 0), (0, 1)],
            (1, 1): [(-1, 0), (0, -1), (0, 0)],
            (-1, -1): [(0, 0), (1, 0), (0, 1)],
            (-1, 0): [(0, -1), (0, 0), (0, 1)],
            (0, -1): [(-1, 0), (0, 0), (1, 0)],
            (1, -1): [(-1, 0), (0, 0), (0, 1)],
            (-1, 1): [(1, 0), (0, 0), (0, -1)]
    }

    link_moves = dict()
    for key, vals in _link_moves.items():
        link_moves[t_encoder[key]] = [t_encoder[v] for v in vals]
    
    normalized = np.array(list(ascending(8, 0, n_links)), dtype=np.ubyte)
    
    r_encoder, r_decoder = rotations_index(n_links)

    if not permutations_path.is_file() or force:
        permutations = permutations_index(n_links)
        np.save(permutations_path, permutations)
    else:
        permutations = np.load(permutations_path)

    unique = set()
    if not encoder_path.is_file() or force:
        encoder = np.zeros(([3] * (2 * n_links) + [2]), dtype=np.uintc)
        for p_idx, p in enumerate(permutations):
            for n_idx, n in enumerate(normalized):
                cfg = np.concatenate([t_decoder[x] for x in n], axis=0)
                p_cfg = cfg[p]
                if not tuple(p_cfg) in unique:
                    encoder[tuple(p_cfg)] = [p_idx, n_idx]
                else: unique.add(tuple(p_cfg))
        np.save(encoder_path, encoder)
    else:
        encoder = np.load(encoder_path)

    if not moves_path.is_file() or force:
        moves = []
        for x in normalized:
            options = [link_moves[d] for d in x]
            _moves = []
            for y in list(itertools.product(*options)):
                r = tuple(np.concatenate([t_decoder[_x] for _x in y], axis=0))
                r_encoded = r_encoder[r]
                _moves.append(r_encoded)
            moves.append(_moves)
        moves = np.array(moves, dtype=np.uintc)
        np.save(moves_path, moves)
    else:
        moves = np.load(moves_path)

    return {
        "encoder": encoder,
        "permutations": permutations,
        "rotations": r_decoder,
        "moves": moves,
    }


def normalize(cfg, lengths):
    normed = cfg.reshape(-1,2) // lengths.reshape(-1,1)
    return normed.reshape(-1)


def updates(cfg, lengths, moves, encoder, rotations, permutations):
    base = normalize(cfg, lengths)
    _base = encoder[tuple(base)]
    perm = permutations[_base[0]]

    return rotations[moves[_base[1]]][:, perm]
    

if __name__ == "__main__":
    n_links = 8
    lengths = np.array(links(n_links), dtype=np.byte)
    cfg = np.array(start_cfg(n_links),  dtype=np.byte)

    data = gen_data(n_links, Path("."), force=False)

    moves = data["moves"]
    rotations = data["rotations"]
    permutations = data["permutations"]
    encoder = data["encoder"]

    link_rotations = updates(
        cfg=cfg, 
        lengths=lengths, 
        moves=moves, 
        encoder=encoder, 
        rotations=rotations, 
        permutations=permutations
    )