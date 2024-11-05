import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import bisect


def split_data(df, ratio=0.8):
    # we just need to shuffle the data before proceeding, and reset the index
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # we're getting the ratio
    index = int(df.shape[0] * ratio)

    # so here we're splitting the data into test and train using our index variable
    train_df = shuffled_df.iloc[:index]
    test_df = shuffled_df.iloc[index:]

    # and we're getting the data used to predict and dropping it from X_train and X_test
    y_train = train_df['price']
    y_test = test_df['price']
    X_train = train_df.drop(columns=['price', 'price^2'], axis=1)
    X_test = test_df.drop(columns=['price', 'price^2'], axis=1)

    return X_train, y_train, X_test, y_test


def polynomial_features(df, degree=2):
    numerical_columns = df.select_dtypes(include=['int64']).columns

    # Create polynomial features up to the specified degre
    for col in numerical_columns:
        for d in range(2, degree + 1):
            new_col_name = f"{col}^" + str(d)
            df[new_col_name] = df[col] ** d
    return df


def scale_data(df):
    # Select numerical columns as we added some
    numerical_columns = df.select_dtypes(include=['int64']).columns

    # Calculate mean and standard deviation for scaling
    for col in numerical_columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    return df


def label_encoder_object(df):
    categorical_columns = df.select_dtypes(include=['object', 'datetime']).columns

    for col in categorical_columns:
        unique_values = df[col].unique()
        encoding_map = {value: label for label, value in enumerate(unique_values)}
        df[col] = df[col].map(encoding_map)
    return df


def one_hot_encoder(df):
    boolean_columns = df.select_dtypes(include=['bool']).columns

    for col in boolean_columns:
        df[col] = df[col].apply(lambda x: 1 if x == True else 0)
    return df


def remove_unimportant(df):
    df.reset_index(drop=True, inplace=True)
    df.drop('maker_key', axis=1, inplace=True)
    df.drop('model_key', axis=1, inplace=True)
    df.drop('registration_date', axis=1, inplace=True)
    df.drop('sold_at', axis=1, inplace=True)
    return df


def linear_regression(X_train, y_train, X_test, y_test):
    linearRegression = LinearRegression()

    # fit the model to the training data
    linearRegression.fit(X_train, y_train)

    # use the model to predict on the test set
    y_pred = linearRegression.predict(X_test)

    # evaluate the model using r2_score
    print("R squared score of the linear regression model: ", r2_score(y_test, y_pred))

    coefs = pd.DataFrame(
        linearRegression.coef_, columns=["Coefficients"], index=X_train.columns
    )
    coefs.plot(kind="barh", figsize=(9, 7))
    plt.title("Linear regression")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()
    return coefs


def ridge_regression(X_train, y_train, X_test, y_test):
    # Define a list of alpha values to consider
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 71, 100, 1000]

    # Create and fit the RidgeCV model
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train, y_train)

    # Print the best alpha value
    print(f"Best alpha value found by RidgeCV: {ridge_cv.alpha_}")

    # Evaluate on the test set
    y_pred = ridge_cv.predict(X_test)
    print("R² score:", r2_score(y_test, y_pred))

    coefs = pd.DataFrame(
        ridge_cv.coef_, columns=["Coefficients"], index=X_train.columns
    )
    coefs.plot(kind="barh", figsize=(9, 7))
    plt.title("Ridge Linear regression")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()
    return coefs


def lasso_regression(X_train, y_train, X_test, y_test):
    # define a list of alpha
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # create and fit the RidgeCV regression model
    lasso_cv = LassoCV(alphas=alphas, cv=5)
    lasso_cv.fit(X_train, y_train)

    # Print the best alpha value that we got from fitting ridge_cv
    print(f"Best alpha value found by Lasso: {lasso_cv.alpha_}")

    # use the model to predict on the test set
    y_pred_lasso = lasso_cv.predict(X_test)

    # evaluate the model using r2_score
    print("R squared score of lasso model: ", r2_score(y_test, y_pred_lasso))

    coefs = pd.DataFrame(
        lasso_cv.coef_, columns=["Coefficients"], index=X_train.columns
    )
    coefs.plot(kind="barh", figsize=(9, 7))
    plt.title("Lasso Linear regression")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()
    return coefs


def plot_feature_importance(model_coefs):
    plt.figure(figsize=(15, 8))
    for name, coefs in model_coefs.items():
        coefs.plot(kind="barh", figsize=(9, 7))
        plt.title("{name}'s feature importance".format(name=name))
        plt.axvline(x=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.show()
    plt.tight_layout()
    plt.show()


df = pd.read_csv("./data/bmw_pricing_challenge.csv")

df = polynomial_features(df)
df = scale_data(df)
df = remove_unimportant(df)
df = label_encoder_object(df)
df = one_hot_encoder(df)
X_train, y_train, X_test, y_test = split_data(df)

linearRegression_coefs = linear_regression(X_train, y_train, X_test, y_test)

ridge_coefs = ridge_regression(X_train, y_train, X_test, y_test)

lasso_coefs = lasso_regression(X_train, y_train, X_test, y_test)

model_coefs = {'LinearRegression': linearRegression_coefs,
               'LassoCV': lasso_coefs,
               'RidgeCV': ridge_coefs}
plot_feature_importance(model_coefs)
