import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Optional


def print_correlation_matrix(df: pd.DataFrame, description: Optional[str] = None,
                             correlation_param: str = "stringency_index") -> None:

    """
    Given a pandas dataframe, a correlation variable and a description, print a sorted correlation matrix for how all
    variables correlate to the given correlation parameter (by default 'stringency_index').

    :param df: dataframe with covid data for which we want correlation matrix
    :param correlation_param: str the parameter for which we want to find correlations
    :param description:
    :return: None
    """
    print(f"Correlation matrix {description}")
    stringency_corr_mat = df.corr("pearson")
    sorted_mat = stringency_corr_mat.unstack().sort_values()[correlation_param]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(sorted_mat)


def print_df_size(df: pd.DataFrame, description: Optional[str] = None) -> None:
    """
    Given a pandas dataframe and a description, print its size (number of records).

    :param df: dataframe whose size we want to print
    :param description: string description of dataframe to be printed
    :return: None
    """
    print(f"\nDataframe {description} has size {df.size}\n")


def analyze_data(filepath: str, country_name: str, start_date: str = "2020-01-01",
                 end_date: str = "2022-03-31") -> None:
    """
    Given the filepath to the covid data csv file, country of interest and  run linear regressions.

    :param filepath: path to the csv file
    :param country_name: name of country we are interested in
    :param start_date: str in ISO format representing start of time range we are interested in
    :param end_date: str in ISO format representing end of time range we are interested in
    :return: None
    """
    # read full dataframe with data for all countries, in full time range
    covid_data = pd.read_csv(filepath)
    # convert 'date' column to datetime format, so we can filter by date later
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    # show all columns of dataframe
    with pd.option_context("display.max_columns", None):
        print(covid_data.head())
    print_df_size(covid_data, "all countries")
    # filter dataframe for given country
    covid_data_country = covid_data.query(f"location == '{country_name}'")
    # raise error if dataframe is empty after querying
    if covid_data_country.size < 1:
        raise ValueError("No data exists for given country")
    print_df_size(covid_data_country, country_name)
    print_correlation_matrix(covid_data_country, f"{country_name} all time")
    # convert start date and end date into datetime objects so we can use them to filter
    start_date: datetime = datetime.fromisoformat(start_date)
    end_date: datetime = datetime.fromisoformat(end_date)
    # time range query
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] > start_date) & (covid_data_country['date'] < end_date)]
    print_df_size(covid_data_country_timeframe, f"{country_name} from {start_date} to {end_date}")
    print_correlation_matrix(covid_data_country_timeframe, f"{country_name} from {start_date} to {end_date}")
    save_scatter("icu_patients", covid_data_country_timeframe)


def save_scatter(independent_variable: str, df: pd.DataFrame) -> None:
    """
    Given an independent variable (such as 'new_cases'), and a dataframe, see how it affects the stringency index
    using linear regression, and display a basic scatter plot.
    
    :param independent_variable: str name of variable whose effect we are trying to quantify
    :param df: dataframe containing data to be plotted
    :return: None
    """
    df = df[["stringency_index", independent_variable]].dropna()
    x = df[independent_variable]
    y = df["stringency_index"]
    plt.scatter(x=x, y=y)
    plt.savefig("scatter.png")
    # get equation of best fit lineq
    reg = np.polyfit(x, y, deg=1)
    print(f"equation of trendline is given by: {reg}")
    trend = np.polyval(reg, x)
    plt.scatter(x=x, y=y)
    plt.plot(x, trend, 'r')
    plt.savefig("scatter_trendline")





