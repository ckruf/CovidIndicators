import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
from typing import Optional, List


def correlation_matrix(df: pd.DataFrame, description: Optional[str] = None,
                       correlation_param: str = "stringency_index", min_correlation: float = 0.45) -> List[str]:
    """
    Given a pandas dataframe, a correlation variable and a description, print a sorted correlation matrix for how all
    variables correlate to the given correlation parameter (by default 'stringency_index'). Returns a list of parameters
    whose (absolute) R value is greater than the min_correlation value (defaults to 0.5).

    :param df: dataframe with covid data for which we want correlation matrix
    :param description:
    :param correlation_param: str the parameter for which we want to find correlations
    :param min_correlation: the function will return all parameters whose correlation is greater than this value

    :return: None
    """
    print(f"Correlation matrix {description}\n")
    stringency_corr_mat = df.corr("pearson")
    sorted_mat = stringency_corr_mat.unstack().sort_values()[correlation_param]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(sorted_mat)
    filtered_mat = sorted_mat.dropna()
    filtered_mat = filtered_mat.to_frame()
    filtered_mat = filtered_mat[abs(filtered_mat[0]) > min_correlation]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(filtered_mat)
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
                                    description=f"{country_name} from {start_date} to {end_date}",
                                    min_correlation=min_correlation)
    # filter correlated params to remove 'per_thousand' and 'per_million', because within
    # same country, absolute quantity is enough
    correlated = list(filter(lambda x: "per_thousand" and "per_million" not in x, correlated))
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


def plot_scatter(independent_variable: str, df: pd.DataFrame, country_name: str,
                 start_date: str, end_date: str) -> None:
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
    df = df[["stringency_index", independent_variable]].dropna()
    if df.size < 5:
        raise ValueError("Fewer than 5 records, no point in scatter")
    x = df[independent_variable]
    y = df["stringency_index"]
    plt.scatter(x=x, y=y)
    # add title and labels for axes
    plt.suptitle(f"{country_name} between {start_date} and {end_date}")
    plt.xlabel(independent_variable)
    plt.ylabel("stringency index")
    # save as image
    plt.savefig(f"scatter_{independent_variable}_{country_name}.png")


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


def multiple_linear_regression():
    pass
