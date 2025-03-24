""" Run this script or `pytest -s -v` from the sapicore source root directory to execute all test configurations.
Select or exclude marked tests by category with `-m` (unit, integration, functional, slow).

This script generates an HTML coverage report under tests/coverage_report.
"""
import os
import pytest

from coverage import Coverage
from argparse import ArgumentParser

from sapicore.utils.io import ensure_dir
from sapicore.tests import ROOT


def run_tests(root=ROOT):
    # parse runtime arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        action="store",
        dest="m",
        nargs="+",
        default="all",
        help="Test categories to run (unit, integration, functional, slow, or all).",
    )

    parser.add_argument("-v", action="store_true", dest="v", default=False, help="Verbose pytest output.")
    parser.add_argument("-s", action="store_true", dest="s", default=False, help="Show runtime output, no capture.")

    # parse arguments to pass along to pytest, e.g. test category selection.
    args = parser.parse_args()

    args_list = []
    if args.v:
        args_list += ["-v"]

    if args.s:
        args_list += ["-s"]

    if args.m and "all" not in args.m:
        args_list += ["-m " + " ".join(args.m)]

    # unavoidable--pytest refuses to work with .ini at root or accept absolute paths when executed from here.
    os.chdir(root)

    # initialize coverage.py object.
    cov = Coverage(omit=[os.path.join(root, folder, "*") for folder in ["scripts"]])
    cov.start()

    # run configurable tests.
    pytest.main(args=args_list)

    # generate HTML coverage report.
    cov.stop()
    cov.html_report(directory=ensure_dir(os.path.join(root, "tests", "scripts", "coverage_report")))


if __name__ == "__main__":
    run_tests()
