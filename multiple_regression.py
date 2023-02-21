from linear_algebra import Vector, Matrix


class MultipleRegression:
    """Multiple Regression model"""

    def __init__(self):
        self.coef_no = 3                                # Number of coefficients
        self.beta = [0 for _ in range(self.coef_no)]    # Initial vector of coefficients
        self.epochs = 1000                              # Number of epochs

    def predict(self, x: Matrix) -> Vector:
        """y_i = B_0 + B_1 * x_i1 + ... + B_n * x_in"""
        return [sum(x_i * b_i for x_i, b_i in zip(x[i], self.beta)) for i in range(len(x))]

    def error(self, x: Matrix, y: Vector) -> Vector:
        """e_i = y_i - (B_0 + B_1 * x_i1 + ... + B_n * x_in)"""
        return [y_i - pred_y for y_i, pred_y in zip(y, self.predict(x))]

    def train(self, x: Matrix, y: Vector):
        """Gradient Descent optimization"""
        for _ in range(self.epochs):
            pass
