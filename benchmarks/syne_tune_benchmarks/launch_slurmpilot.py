import os
from argparse import ArgumentParser
from pathlib import Path

from slurmpilot import SlurmPilot, JobCreationInfo
from slurmpilot.config import load_config
from slurmpilot.util import unify
from tqdm import tqdm

from baselines import (
    Methods,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=False, default="bench")
    parser.add_argument("--n_workers", type=int, required=False, default=1)
    parser.add_argument("--num_seeds", type=int, required=False, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--checkpoint_dir", type=str, required=False, default="")
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument(
        "--slurmpilot_folder", type=str, required=False, default="~/slurmpilot/jobs"
    )
    parser.add_argument(
        "--run_hpob_only",
        action='store_true',
        help="If set, only runs hpobench benchmarks, otherwise runs all benchmarks.",
    )
    parser.add_argument(
        "--run_tabrepo_only",
        action='store_true',
        help="If set, only runs tabrepo benchmarks, otherwise runs all benchmarks.",
    )
    parser.add_argument("--sbatch_arguments", type=str, required=False)

    args, _ = parser.parse_known_args()

    experiment_tag = args.experiment_tag

    num_seeds = args.num_seeds
    cluster = args.cluster
    partition = args.partition
    sbatch_arguments = args.sbatch_arguments
    slurmpilot_folder = args.slurmpilot_folder


    if args.run_hpob_only:
        from hpob_benchmarks import hpob_benchmark_definitions
        benchmarks_selected  =  list(hpob_benchmark_definitions.keys())
    elif args.run_tabrepo_only:
        from tabrepo_benchmarks import tabrepo_benchmark_definitions
        benchmarks_selected = list(tabrepo_benchmark_definitions.keys())
    else:
        from blackbox_benchmarks import benchmark_definitions
        benchmarks_selected =  list(benchmark_definitions.keys())

    methods_selected = [
        Methods.RS,
#        Methods.OPT_REA,
#        Methods.OPT_RS,
        Methods.REA,
        Methods.TPE,
        Methods.BORE,
        Methods.CQR,
#        Methods.OriginalOptFormerGPUCB,
#        Methods.OriginalOptFormerRS,
#        Methods.OriginalOptFormerHillClimb
    ]
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    config = load_config()

    slurm = SlurmPilot(config=config, clusters=[cluster], ssh_engine="ssh")
    max_runtime_minutes = 60 * 24 - 1
    python_args = []
    for method in tqdm(methods_selected):
        for benchmark in benchmarks_selected:
            # To avoid memory issues, we run only one seed per job
            for seed in range(num_seeds):
                python_args.append(
                    {
                        "method": method,
                        "n_workers": args.n_workers,
                        "benchmark": benchmark,
                        "checkpoint_dir": args.checkpoint_dir,
                        # run all seeds in [0, seed-1]
                        "seed": seed,
                        "run_all_seed": 0,
                    }
                )

    jobname = unify(f"synetune/{experiment_tag}", method="date")

    print(f"Going to launch {len(python_args)} jobs, jobname: {jobname}")
    jobinfo = JobCreationInfo(
        cluster=cluster,
        partition=partition,
        sbatch_arguments=sbatch_arguments,
        jobname=jobname,
        entrypoint="benchmark_main.py",
        python_args=python_args,
        src_dir=str(Path(__file__).parent),
        python_binary="python",
        python_libraries=[
            str(Path(__file__).parent.parent.parent / "open_optformer"),
        ],
        n_cpus=4,
        n_gpus=0 if cluster == 'barnard' else 1,  # barnard has no GPU
        mem=1024 * 128 if cluster == 'barnard' else 1024 * 16,
        nodes=1,
        max_runtime_minutes=max_runtime_minutes,
        bash_setup_command="source ~/.bashrc; conda activate optformer",
        env={
            # write tuner files in Slurmpilot folder corresponding to `jobname`
            "SYNETUNE_FOLDER": f"slurmpilot/jobs/{jobname}",
        },
        n_concurrent_jobs=20,  # max number of jobs to run at the same time, setting this number to high will lead to throttling by huggingface
    )
    if not args.dry_run:
        jobid = slurm.schedule_job(jobinfo)
