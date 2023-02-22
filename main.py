import pandas as pd
from data_visualization import regplot
from linear_regression import LinearRegression
from multiple_regression import MultipleRegression
from linear_algebra import create_extended_matrix


if __name__ == '__main__':

    # Preparing data for our model
    raw_data = pd.read_csv('real_estate_price_size_year.csv')
    size = raw_data['size'].values.tolist()
    price = raw_data['price'].values.tolist()
    year = raw_data['year'].values.tolist()

    # Linear Regression model
    lr = LinearRegression()
    alpha, beta = lr.train(size, price)

    # Multiple Regression model
    mr = MultipleRegression()
    features = create_extended_matrix([size, year])
    mr.train(features, price)

    # Results
    regplot(size, price, alpha, beta)
    print(f"r_squared = {lr.r_squared(size, price)}")
    print(f"multiple_r_squared = {mr.r_squared(features, price)}")
