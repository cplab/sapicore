""" Starts a tensorboard server with log directory set to module/script/run, given as runtime arguments.

Modules are folders containing test scripts (e.g., network, ensemble). Scripts are test scripts (e.g., test_network).
If called without -script, "test_"+module will be used by default.

Runs are individual timestamped directories within the test script directory, given as integer indices where zero
indicates the earliest run found. If called with -run -1 or without -run, the most recent run will be used.

The following calls from sapicore/tests/scripts are equivalent, reflecting the above assumptions:

* ``python -m start_tb -module network -script test_network -run -1``
* ``python -m start_tb -module network -script test_network``
* ``python -m start_tb -module network``

User may view tensorboard content by opening a browser at localhost:6006.
"""
import os
import sys
import logging

from argparse import ArgumentParser
from sapicore.utils.io import log_settings


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-module",
        action="store",
        nargs="?",
        type=str,
        dest="module",
        help="test module directory name.",
    )
    parser.add_argument(
        "-script", action="store", type=str, nargs="?", dest="script", help="test script directory name.", default=None
    )
    parser.add_argument(
        "-run",
        action="store",
        nargs="?",
        type=int,
        dest="run",
        help="run directory serial id. Most recent one loaded by default.",
        default=-1,
    )
    args = parser.parse_args()
    if not args.script:
        args.script = "test_" + args.module

    test_root = os.path.join(os.path.dirname(sys.argv[0]), "..", "engine", args.module, args.script)
    dirs = os.listdir(test_root)

    run_dirs = []
    for item in dirs:
        if item.split("-")[0].isnumeric():
            run_dirs.append(item)

    # initialize default logger.
    log_settings()

    if len(run_dirs) == 0:
        logging.info(f"No runs found at {args.module}/{args.script}. Terminating.")
    elif len(run_dirs) <= args.run:
        logging.info(f"Run ID given ({args.run}) is out of range (0-{len(run_dirs)}). Terminating.")
    else:
        # start tensorboard server at selected log directory.
        run_dirs = sorted(run_dirs)
        logging.info(f"Starting tensorboard server at {os.path.realpath(os.path.join(test_root, run_dirs[args.run]))}")

        log_dir = os.path.join(test_root, run_dirs[args.run], "tb")
        os.system(f"tensorboard --logdir={log_dir}")
