from linear_algebra import Vector


def mean(x: Vector) -> float:
    """Simple average of the dataset"""
    return sum(x) / len(x)
