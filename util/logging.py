import logging
import torch.distributed as dist


def setup_root_logger(level=logging.INFO):
    """
    Configure the root logger to only log from rank 0 process.
    This allows using simple logging.info() calls throughout the code.
    """
    # Get the rank (default to 0 if distributed not initialized)
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    # Clear any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Set logging level
    root.setLevel(level)

    # For rank 0, add a normal handler
    if rank == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%m-%d,%H:%M'
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        # For other ranks, set a very high logging level to suppress output
        root.setLevel(logging.CRITICAL + 1)  # Higher than any standard level

