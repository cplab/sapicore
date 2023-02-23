""" Run this script to purge old run output and caches recursively in all test script directories. """
import os
import shutil
import logging

from sapicore.utils.io import log_settings


def get_size(start_path: str = "."):
    """Recursively find total size in bytes of files in or below the directory `start_path`."""
    total_size = 0
    for dir_path, dir_names, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dir_path, f)

            # skip if it is symbolic link.
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


if __name__ == "__main__":
    # initialize default logger.
    log_settings()

    # initialize list of directories to remove.
    chopping_block = []

    # if script directory treated as root, go up to project root.
    src_root = os.path.join(os.getcwd(), "..", "..") if "scripts" in os.getcwd() else os.getcwd()

    # walk all directories below project root recursively and delete test run output, pycache, and coverage.
    for root, _, _ in os.walk(src_root):
        items = os.listdir(root)
        for item in items:
            if os.path.basename(root).startswith("test_"):
                if item.split("-")[0].isnumeric():
                    logging.info(f"Deleting run {item} files from {os.path.basename(root)}.")
                    chopping_block.append(os.path.join(root, item))
            else:
                deletion = any([word in item for word in ["coverage", "pytest", "pycache", "hypothesis"]])
                exclusion = ".ini" in item or "coveragerc" in item

                if deletion and not exclusion:
                    # also removes code coverage report and caches in all test subdirectories.
                    logging.info(f"Deleting {item} from {os.path.basename(root)}.")
                    chopping_block.append(os.path.join(root, item))

    # space to be freed in MB.
    space = sum([get_size(item) for item in chopping_block]) / 10**6

    for item in chopping_block:
        try:
            shutil.rmtree(item) if os.path.isdir(item) else os.remove(item)
        except FileNotFoundError:
            pass

    logging.info(f"Freed {space:.2f} MB.")
