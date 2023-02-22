from linear_algebra import Vector, Matrix, get_column
from statistics import mean


class MultipleRegression:
    """Multiple Regression model"""

    def __init__(self):
        self.coef_no = 3                              # Number of coefficients
        self.beta = [0 for _ in range(self.coef_no)]  # Initial vector of coefficients
        self.epochs = 1000
        self.learning_rate = 0.000000001

    def predict(self, x: Matrix) -> Vector:
        """y_i = B_0 + B_1 * x_i1 + ... + B_n * x_in"""
        return [sum(x_i * b_i for x_i, b_i in zip(x[i], self.beta)) for i in range(len(x))]

    def error(self, x: Matrix, y: Vector) -> Vector:
        """e_i = y_i - (B_0 + B_1 * x_i1 + ... + B_n * x_in)"""
        return [y_i - pred_y for y_i, pred_y in zip(y, self.predict(x))]

    def residual_sum_of_squares(self, x: Matrix, y: Vector):
        """Loss function: RSS = sum(e^2)"""
        return sum(e ** 2 for e in self.error(x, y))

    def total_sum_of_squares(self, y: Vector):
        """TSS = sum((y - y_mean)^2)"""
        y_mean = mean(y)
        return sum([(y_i - y_mean) ** 2 for y_i in y])

    def r_squared(self, x: Matrix, y: Vector):
        """R^2 = 1 - RSS/TSS"""
        return 1.0 - self.residual_sum_of_squares(x, y) / self.total_sum_of_squares(y)

    def train(self, x: Matrix, y: Vector):
        """Gradient Descent optimization"""
        for _ in range(self.epochs):

            # Partial derivatives
            err = self.error(x, y)
            gradient = [-2 * sum([e_i * x_i
                                  for e_i, x_i in zip(err, get_column(x, i))])
                        for i in range(self.coef_no)]

            # Moves step_size in the gradient direction
            self.beta = [b + -self.learning_rate * grad for b, grad in zip(self.beta, gradient)]

        return self.beta
