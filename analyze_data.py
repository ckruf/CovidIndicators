import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from datetime import datetime
from typing import Optional, List
from math import sqrt


def correlation_matrix(df: pd.DataFrame, description: Optional[str] = None,
                       correlation_param: str = "stringency_index", min_correlation: float = 0.45) -> List[str]:
    """
    Given a pandas dataframe, a correlation variable and a description, print a sorted correlation matrix for how all
    variables correlate to the given correlation parameter (by default 'stringency_index'). Saves the produced
    correlation matrix as a .csv and returns a list of parameters whose (absolute) Pearson R value is greater than the
    min_correlation value (defaults to 0.5).

    :param df: dataframe with covid data for which we want correlation matrix
    :param description: description of the correlation matrix to be printed to console and used for csv filename
    :param correlation_param: str the parameter for which we want to find correlations
    :param min_correlation: the function will return all parameters whose correlation is greater than this value
    :return: List[str] names of columns for which abs(R-value) > min correlation
    """
    print(f"Correlation matrix {description}\n")
    # get Pearson R values for all variables in the matrix
    stringency_corr_mat = df.corr("pearson")
    sorted_mat = stringency_corr_mat.unstack().sort_values()[correlation_param]
    # print full length
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(sorted_mat)
    # drop NA values
    filtered_mat = sorted_mat.dropna()
    filtered_mat = filtered_mat.to_frame()
    # filter to keep params whose correlation is greater than min correlation
    filtered_mat = filtered_mat[abs(filtered_mat[0]) > min_correlation]
    filename = description.replace(" ", "_")
    filtered_mat.to_csv(f"{filename}.csv")
    print(f"Correlation matrix {description} - NA filtered\n")
    # print filtered correlation matrix
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(filtered_mat)
    # get all highly correlated params, convert to list and return
    rows = filtered_mat.index
    row_list = list(rows)
    return row_list


def print_df_size(df: pd.DataFrame, description: Optional[str] = None) -> None:
    """
    Given a pandas dataframe and a description, print its size (number of records).

    :param df: dataframe whose size we want to print
    :param description: string description of dataframe to be printed
    :return: None
    """
    print(f"\nDataframe {description} has size {df.size}\n")


def analyze_data(filepath: str, country_name: str, start_date: str = "2020-01-01",
                 end_date: str = "2022-03-31", min_correlation: float = 0.45,
                 plot_all: bool = False, plot_all_trendline: bool = False) -> None:
    """
    Given the filepath to the covid data csv file, country of interest and  run linear regressions.

    :param filepath: path to the csv file
    :param country_name: name of country we are interested in
    :param start_date: str in ISO format representing start of time range we are interested in
    :param end_date: str in ISO format representing end of time range we are interested in
    :param min_correlation:
    :param plot_all: if True, scatter plots for all variables with correlation greater than min_correlation
    :param plot_all_trendline: if True, scatter plots w/ trendline for all variables with correlation greater than min_correlation
    :return: None
    """
    # read full dataframe with data for all countries, in full time range
    covid_data = pd.read_csv(filepath)

    # convert 'date' column to datetime format, so we can filter by date later
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # show sample of dataframe and print all columns
    print(covid_data.head())
    column_list = covid_data.columns.values.tolist()
    print(f"All columns: {column_list}")
    print_df_size(covid_data, "all countries")

    # filter dataframe for given country
    covid_data_country = covid_data.query(f"location == '{country_name}'")
    # raise error if dataframe is empty after querying
    if covid_data_country.size < 1:
        raise ValueError("No data exists for given country")
    # show number of data points for country for all time
    print_df_size(covid_data_country, country_name)
    # print correlation matrix for all time
    _: List[str] = correlation_matrix(covid_data_country, f"{country_name} all time")

    # convert start date and end date into datetime objects so we can use them to filter
    start_date_object: datetime = datetime.fromisoformat(start_date)
    end_date_object: datetime = datetime.fromisoformat(end_date)

    # time range query
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] > start_date_object) &
                                                      (covid_data_country['date'] < end_date_object)]

    # show number of data points for given time range
    print_df_size(covid_data_country_timeframe, f"{country_name} from {start_date} to {end_date}")
    # print correlation matrix
    correlated = correlation_matrix(df=covid_data_country_timeframe,
                                    description=f"{country_name} {start_date} to {end_date} corr mat",
                                    min_correlation=min_correlation)
    # filter correlated params to remove 'per_thousand' and 'per_million' params, because within
    # same country, absolute quantity is enough
    correlated = list(filter(lambda x: "per_thousand" not in x and "per_million" not in x, correlated))
    # make scatter for single parameter
    plot_scatter("positive_rate", covid_data_country_timeframe, country_name, start_date, end_date)
    plot_scatter_with_trendline("positive_rate", covid_data_country_timeframe, country_name, start_date, end_date)
    # make scatters for all parameters greater than min_correlation
    if plot_all:
        for param in correlated:
            plot_scatter(param, covid_data_country_timeframe, country_name, start_date, end_date)
    if plot_all_trendline:
        for param in correlated:
            plot_scatter_with_trendline(param, covid_data_country_timeframe, country_name, start_date, end_date)
    multiple_linear_regression(correlated, covid_data_country_timeframe, country_name, start_date, end_date)
    alt_multiple_linear_regression_with_plot(correlated, covid_data_country_timeframe, country_name, start_date, end_date)


def plot_scatter(independent_variable: str, df: pd.DataFrame, country_name: str,
                 start_date: str, end_date: str, dependent_variable: str = "stringency_index") -> None:
    """
    Given an independent variable (such as 'new_cases'), and a dataframe, produce a scatter plot and
    save it as an image.
    
    :param independent_variable: str name of variable whose effect we are trying to quantify
    :param df: dataframe containing data to be plotted
    :param country_name: str country name (to label plot)
    :param start_date: str begin of date range (to label plot)
    :param end_date: str end of date range (to label plot)
    :return: None
    """
    # only get the two columns we are interested in from dataframe and drop NA values
    df = df[[dependent_variable, independent_variable]].dropna()
    if df.size < 5:
        raise ValueError("Fewer than 5 records, no point in scatter")
    x = df[independent_variable]
    y = df[dependent_variable]
    plt.scatter(x=x, y=y)
    # add title and labels for axes
    plt.suptitle(f"{country_name} between {start_date} and {end_date}")
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    # save as image
    plt.savefig(f"scatter_{independent_variable}_{country_name}.png")
    plt.close()


def plot_scatter_with_trendline(independent_variable: str, df: pd.DataFrame, country_name: str,
                                start_date: str, end_date: str) -> None:
    """
    Given an independent variable (such as 'new_cases'), and a dataframe, produce a scatter plot, including
    a trendline, its equation and an R^2 value.

    :param independent_variable: str name of variable whose effect we are trying to quantify
    :param df: dataframe containing data to be plotted
    :param country_name: str country name (to label plot)
    :param start_date: str begin of date range (to label plot)
    :param end_date: str end of date range (to label plot)
    :return: None
    """
    # only get the two columns we are interested in from dataframe and drop NA values
    df = df[["stringency_index", independent_variable]].dropna()
    if df.size < 5:
        raise ValueError("Fewer than 5 records, no point in scatter")
    x = df[independent_variable]
    y = df["stringency_index"]
    # get equation of best fit line
    reg = np.polyfit(x, y, deg=1)
    eqtn = np.poly1d(reg)
    # calculate R^2 value
    x_one_dimensional = x.values.reshape(-1, 1)  # must convert to 1D array for calculation
    model = LinearRegression()
    model.fit(x_one_dimensional, y)
    r_squared = model.score(x_one_dimensional, y)
    # plot graph with trendline and save as image
    trend = np.polyval(reg, x)
    plt.scatter(x=x, y=y)
    plt.suptitle(f"{country_name} between {start_date} and {end_date}")
    plt.title("y=%.3fx+%.3f; R^2 = %.3f" % (eqtn[1], eqtn[0], r_squared))
    plt.xlabel(independent_variable)
    plt.ylabel("stringency index")
    plt.plot(x, trend, 'r')
    plt.savefig(f"scatter_trendline_{independent_variable}_{country_name}.png")
    plt.close()


def multiple_linear_regression(predictors: List[str], df: pd.DataFrame, country_name: str,
                               start_date: str, end_date: str) -> None:
    """
    Given a list of predictors (independent variables) for which we have data in the given data frame, produce
    a multiple linear regression model based on all the given predictors.

    :param predictors: list of str - column names of predictors of 'stringency_index'
    :param df: dataframe containing the data
    :param country_name: str country name
    :param start_date: str begin of date range
    :param end_date: str end of date range
    :return: None
    """
    interested_columns = predictors.copy()
    interested_columns.append("stringency_index")
    # filter only columns we are interested in
    df = df[interested_columns]
    # drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv("df_multiple_regression.csv")
    df = df.dropna()
    df.to_csv("df_multiple_regression_na_dropped.csv")
    x = df[predictors]
    x = x.dropna()
    y = df["stringency_index"]
    # x = np.array(x)
    # y = np.array(y)
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    print(f"R squared of multiple regression model is {r_sq}\n")
    print("Correlation coefficients are: \n")
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(coeff_df)


def alt_multiple_linear_regression_with_plot(predictors: List[str], df: pd.DataFrame, country_name: str,
                                   start_date: str, end_date: str) -> None:
    """
    Given a list of predictors (independent variables) for which we have data in the given data frame, produce
    a multiple linear regression model based on all the given predictors - alternative method.

    :param predictors: list of str - column names of predictors of 'stringency_index'
    :param df: dataframe containing the data
    :param country_name: str country name
    :param start_date: str begin of date range
    :param end_date: str end of date range
    :return: None
    """
    interested_columns = predictors.copy()
    interested_columns.append("stringency_index")
    # filter only columns we are interested in
    df = df[interested_columns]
    # drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv("df_multiple_regression.csv")
    df = df.dropna()
    df.to_csv("df_multiple_regression_na_dropped.csv")
    x = df[predictors]
    x = x.dropna()
    y = df["stringency_index"]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r_squared = r2_score(y_test, y_pred) * 100
    print(f"R squared value is {r_squared}")
    pred_df = pd.DataFrame({'actual_value': y_test, 'predicted_value': y_pred, 'Difference': y_test - y_pred})
    pred_df.to_csv(f"predicted_vs_actual_values_{country_name}.csv")
    pred_df = pred_df[["actual_value", "predicted_value"]]
    pred_df = pred_df.dropna()
    pred_df = pred_df[pred_df.actual_value > 1]
    print("Actual vs predicted values are: \n")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(pred_df)
    plot_scatter("actual_value", pred_df, country_name, start_date, end_date, "predicted_value")
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
    coeff_df.to_csv(f"correlation_coefficients_multiple_regression_{country_name}.csv")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"Root Mean Squared Error: {sqrt(mean_squared_error(y_test, y_pred))}")



