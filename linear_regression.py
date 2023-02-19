from linear_algebra import Vector
from statistics import mean


class LinearRegression:
    """Linear Regression model"""

    def __init__(self):
        self.alpha = 0
        self.beta = 0
        self.step_size = 0.000000001
        self.epochs = 1000000

    def predict(self, x_i: float):
        """y = beta * x + alpha"""
        return self.beta * x_i + self.alpha

    def error(self, x_i: float, y_i: float):
        """e = beta * x + alpha - y"""
        return self.predict(x_i) - y_i

    def residual_sum_of_squares(self, x: Vector, y: Vector):
        """Loss function: RSS = sum(e^2)"""
        return sum([self.error(x_i, y_i) ** 2 for x_i, y_i in zip(x, y)])

    def total_sum_of_squares(self, y: Vector):
        """TSS = sum((y - y_mean)^2)"""
        y_mean = mean(y)
        return sum([(y_i - y_mean) ** 2 for y_i in y])

    def r_squared(self, x: Vector, y: Vector):
        """R^2 = 1 - RSS/TSS"""
        return 1.0 - self.residual_sum_of_squares(x, y) / self.total_sum_of_squares(y)

    def train(self, x: Vector, y: Vector):
        """Gradient Descent optimization"""
        for _ in range(self.epochs):

            # Partial derivatives
            # 2 * sum(alpha * x + beta - y)
            alpha_derivative = 2 * sum([self.error(x_i, y_i) for x_i, y_i in zip(x, y)])

            # 2 * sum(alpha * x + beta - y) * x
            beta_derivative = 2 * sum([self.error(x_i, y_i) * x_i for x_i, y_i in zip(x, y)])

            # Moves step_size in the gradient direction
            self.alpha = self.alpha + (-self.step_size * alpha_derivative)
            self.beta = self.beta + (-self.step_size * beta_derivative)

        return self.alpha, self.beta
