import pandas as pd
from data_visualization import regplot
from linear_regression import LinearRegression


if __name__ == '__main__':

    # Preparing data for our model
    raw_data = pd.read_csv('real_estate_price_size.csv')
    size = raw_data['size'].values.tolist()
    price = raw_data['price'].values.tolist()

    # Linear Regression model
    lr = LinearRegression()
    alpha, beta = lr.train(size, price)

    # Results
    regplot(size, price, alpha, beta)
    print(f"r_squared = {lr.r_squared(size, price)}")
