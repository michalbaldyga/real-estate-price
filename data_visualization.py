import matplotlib.pyplot as plt
from linear_algebra import Vector


def regplot(xs: Vector, ys: Vector, alpha: float, beta: float) -> None:
    """Linear Regression Plot"""

    # Adding data points to plot
    plt.scatter(xs, ys)

    # Fitting a line to the data
    plt.plot(xs, [alpha + beta * x for x in xs], color="r")

    plt.title(f"y = {round(beta, 3)} * x + {round(alpha, 3)}")
    plt.xlabel("Size")
    plt.ylabel("Price")
    plt.savefig("regplot")
    plt.show()
