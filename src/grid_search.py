import argparse
import json
import time
from pathlib import Path

import numpy as np
import ray
from ray import air, tune
from ray.air import session

from seed import search
from utils import load_data


def run(params, data):

    threshold = params["threshold"]
    best = threshold
    min_visits= int(np.prod(data["image"].shape[:2]))
    n_links = data["n_links"]

    start_time = time.time()
    window = start_time
    timeout = params["timeout"]
    
    _search = search(
        start=data["start"], 
        neighborhoods=data["neighborhoods"], 
        costs=data["costs"], 
        coords=data["coordinates"],
        cfg_path=[data["start"]],
        px_path=[tuple(data["coordinates"][data["start"]])],
        min_visits=min_visits,
        min_loop_size=params["min_loop_size"], 
        max_steps=params["max_steps"], 
        max_degree=params["max_degree"], 
        cost=float(0), 
        threshold=threshold, 
        start_time=start_time, 
        timeout=timeout, 
        seed=params["seed"]
    )

    best_path = None
    for idx, (cfg_path, visited, cost) in enumerate(_search):
        if len(visited) >= min_visits:
            best = min(best, cost)
            if best == cost:
                best_path = np.copy(cfg_path)
                window = time.time()
        if time.time() - window >= timeout:
            break
    
    if best_path is not None:
        trial_dir = Path(session.get_trial_dir())
        results_path = trial_dir / f"{n_links}-path-{int(best // 1)}.npy"
        np.save(results_path, best_path)
    
    tune.report(cost=best)


def setup(n_links, src_path, sweep_path, output_dir):
    data = load_data(n_links, src_path, output_dir)
    data["n_links"] = n_links
    
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    search_space = load_search(sweep_path)
    
    return data, search_space, results_dir


def load_search(sweep_path):
    grid = {}
    with sweep_path.open(mode="r") as f:
        params = json.load(f)
        for param, vals in params.items():
            grid[param] = tune.grid_search(vals)
    
    return grid


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n-links", type=int)
    parser.add_argument("--src-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--sweep-dir", type=str, default="sweeps")
    parser.add_argument("--sweep", type=str)
    parser.add_argument('--smoke-test', default=False, action='store_true')

    args = parser.parse_args()

    n_links = max(args.n_links, 1)
    output_dir = Path(args.output_dir)
    src_path = Path(args.src_path)
    sweep_dir = Path(args.sweep_dir)
    sweep = args.sweep
    smoke_test = args.smoke_test

    if smoke_test:
        sweep = "smoke-test"

    sweep_path = sweep_dir / f"{n_links}-{sweep}.json"

    data, sweep, results_dir = setup(n_links, src_path, sweep_path, output_dir)
    
    tune_config = tune.TuneConfig(
                        metric="cost",
                        mode="min"
                    )

    progress_reporter = ray.tune.CLIReporter(
                            max_report_frequency=10,
                            print_intermediate_tables=False
                        )

    run_config = air.RunConfig(
                    local_dir=results_dir, 
                    name="grid-search",
                    log_to_file=False,
                    verbose=1,
                    progress_reporter=progress_reporter
                )
    
    tuner = tune.Tuner(
        tune.with_parameters(run, data=data),
        param_space=sweep,
        tune_config=tune_config,
        run_config=run_config,
    )
    
    tuner.fit()
