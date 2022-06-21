import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
import calendar
from typing import Optional, List, Union, Tuple
from math import sqrt
import os
from pathlib import Path

from app_logging import log


def filter_dataframe_country_timeframe(df: pd.DataFrame, country: str, start_date: datetime,
                                       end_date: datetime = None) -> pd.DataFrame:
    """
    Given the dataframe containing covid data for all countries for all time, return dataframe filtered for
    the given country, and also given time frame, if start date and end date provided.

    :param df: Dataframe containing data for all countries for all time
    :param start_date: str ISO date marking beginning of time range
    :param end_date: str ISO date marking end of time range
    :param country: str country name we are interested in
    :return: filtered dataframe for given country in given time range
    """
    # filter for country
    covid_data_country = df.query(f"location == '{country}'")
    # raise error if dataframe is empty after querying
    if covid_data_country.size < 1:
        raise ValueError(f"No data exists for given country - {country} - either misspelled or non-existent.")
    print_df_size(covid_data_country, f"{country} all time has size")
    if start_date is None or end_date is None:
        return covid_data_country
    # time range query
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] > start_date) &
                                                      (covid_data_country['date'] < end_date)]
    if covid_data_country_timeframe.size < 1:
        raise ValueError(f"No data exists for {country} between dates {start_date} and {end_date}")
    print_df_size(covid_data_country_timeframe, f"{country} between {start_date} and {end_date}")

    return covid_data_country_timeframe


def correlation_matrix(df: pd.DataFrame,
                       description: Optional[str] = None,
                       correlation_param: str = "stringency_index",
                       min_correlation: float = 0.45,
                       save: bool = True,
                       print_data: bool = True,
                       country_path: Optional[Union[Path, str]] = None) -> List[str]:
    """
    Given a pandas dataframe, a correlation variable and a description, print a sorted correlation matrix for how all
    variables correlate to the given correlation parameter (by default 'stringency_index'). Saves the produced
    correlation matrix as a .csv and returns a list of parameters whose (absolute) Pearson R value is greater than the
    min_correlation value (defaults to 0.45).

    :param df: dataframe with covid data for which we want correlation matrix
    :param description: description of the correlation matrix to be printed to console and used for csv filename
    :param correlation_param: str the parameter for which we want to find correlations
    :param min_correlation: the function will return all parameters whose correlation is greater than this value
    :param save: bool indicator save correlation matrix to .csv?
    :param print_data: bool indicator print correlation matrix?
    :param country_path: path to folder with results for given country
    :return: List[str] names of columns for which abs(R-value) > min correlation
    """
    # create folder
    folder_path = os.path.join(country_path, "correlation_matrices")
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print("Folder already exists, continuing")
    print(f"Finding correlation matrix {description}\n")
    # get Pearson R values for all variables in the matrix
    stringency_corr_mat = df.corr("pearson")
    sorted_mat = stringency_corr_mat.unstack().sort_values()[correlation_param]
    # print full length
    if save:
        filepath = os.path.join(country_path, "correlation_matrices", "full_correlation_matrix.csv")
        sorted_mat.to_csv(filepath)
    # drop NA values
    filtered_mat = sorted_mat.dropna()
    filtered_mat = filtered_mat.to_frame()
    # filter to keep params whose correlation is greater than min correlation
    filtered_mat = filtered_mat[abs(filtered_mat[0]) > min_correlation]
    filename = description.replace(" ", "_")
    if save and min_correlation > 0:
        filepath = os.path.join(country_path, "correlation_matrices", "filtered_correlation_matrix.csv")
        filtered_mat.to_csv(filepath)
    if print_data:
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


def analyze_data(filepath_covid_data: str,
                 countries_list: List[str],
                 parameters_of_interest: List[str],
                 start_date: str = "2020-01-01",
                 end_date: str = "2022-03-31",
                 target_folder: Optional[str] = None) -> None:
    """
    Given the filepath to the covid data csv file, country of interest and  run linear regressions, multiple linear
    regressions and plot graphs (behaviour can be modified using bool switches).

    Data produced includes:
    - correlation matrix for the given country for all time (saved as csv file)
    - correlation matrix for the given country for the given time range (saved as csv file)
    - if plot_all, scatter plots without trendlines for all variables whose Pearson correlation (R value)
    to 'stringency_index' is greater than the min_correlation (saved as png files)
    - if plot_all_trendline, scatter plots with trendlines same as above
    - if multiple_regression, the data used for the multiple linear regression as a csv file, R^2 value of the
    multiple regression (in terminal output) and correlation coefficients (in terminal output)
    - if multiple_regression_alt_trendline a predictive multiple regression (different method) is done, producing
    the data used for the multiple regression (as a csv file), the R^2 value for the multiple regression (to terminal),
    predicted values of stringency index from the multiple regression model compared to actual values of stringency
    index (as a csv file), a scatter plot of predicted vs actual values (without trendline, as .png), correlation
    coefficients (as a csv file) and errors in the predicted model compared to actual values (to terminal)
    - if drop per, then 'per_million' and 'per_thousand' data is not included in plots

    :param filepath_covid_data: path to the owid_covid_data.csv file
    :param countries_list: list of country names which we are interested in
    :param parameters_of_interest: list of porameters we are interested in
    :param start_date: str in ISO format representing start of time range we are interested in
    :param end_date: str in ISO format representing end of time range we are interested in
    :param target_folder: path to folder in which to create Results folder.
    :return: None
    """
    if target_folder is None:
        # get current working directory
        pwd = os.path.dirname(os.path.realpath(__file__))
        try:
            path = os.path.join(pwd, "Results")
            target_folder_path = Path(path)
            os.mkdir(path)
        except FileExistsError:
            print("The Results directory already exists, continuing")
        except FileNotFoundError:
            print("The provided path does not exist, please provide correct path")
            return
    else:
        try:
            target_folder_path: Path = Path(target_folder)
            os.mkdir(target_folder_path)
        except Exception as e:
            print("The provided path is not valid")
            raise e

    # read full dataframe with data for all countries, in full time range
    covid_data = pd.read_csv(filepath_covid_data)
    # convert 'date' column to datetime format, so we can filter by date later
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    for country in countries_list:
        # round dates to first day in start date month and last day in end date month
        start_date_object = datetime.fromisoformat(start_date)
        floored_start_date = start_date_object.replace(day=1)
        end_date_object = datetime.fromisoformat(end_date)
        final_day = calendar.monthrange(end_date_object.year, end_date_object.month)[1]
        ceilinged_end_date = end_date_object.replace(day=final_day)

        find_monthly_correlations(covid_data=covid_data,
                                  country_name=country,
                                  parameters_of_interest=parameters_of_interest,
                                  start_date=start_date,
                                  end_date=end_date,
                                  target_folder_path=target_folder_path)


def find_months(start_month: int, start_year: int, end_month: int, end_year: int) -> List[Tuple[int, int]]:
    """
    Given a start month and start year and an end month and end year, find all months in the given range (inclusive).
    Returns a list of tuples of the form [(11, 2020), (12, 2020), (1, 2021)]

    :param start_month: month of range beginning
    :param start_year: year of range beginning
    :param end_month: month of range end
    :param end_year: year of range end
    :return: list of tuples with the month and the year
    """
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    return [(d.month, d.year) for d in rrule(MONTHLY, dtstart=start, until=end)]


def find_monthly_correlations(covid_data: pd.DataFrame,
                              country_name: str,
                              parameters_of_interest: List[str],
                              start_date: str,
                              end_date: str,
                              target_folder_path: Path) -> None:
    """
    Find correlations between parameters of interest and stringency index for one country for each whole month
    between start date and end date.

    :param covid_data: pandas dataframe containing all covid data
    :param country_name: name of country we are interested in
    :param parameters_of_interest: list of str parameters we are interested in
    :param start_date: beginning of month
    :param end_date: end of month
    :param target_folder_path: folder to save results in
    :return:
    """
    # create folder for country
    country_path = os.path.join(target_folder_path, country_name)
    try:
        os.mkdir(country_path)
    except FileExistsError:
        print("Country folder already exists, continuing")

    raw_data_country_path = os.path.join(country_path, "raw_data")
    try:
        os.mkdir(raw_data_country_path)
    except FileExistsError:
        print("Raw data folder for country already exists, continuing")

    # show sample of dataframe and print all columns
    print(covid_data.head())
    column_list = covid_data.columns.values.tolist()
    print(f"All columns: {column_list}")
    print_df_size(covid_data, "all countries")

    start_date_object = datetime.fromisoformat(start_date)
    end_date_object = datetime.fromisoformat(end_date)
    months = find_months(start_date_object.month, start_date_object.year, end_date_object.month,
                         end_date_object.year)

    df = pd.DataFrame(columns=["country", "range_start", "range_end", "indicator", "correlation_to_stringency"])
    for month, year in months:
        month_start_date_object = datetime(year, month, 1)
        final_day_in_month = calendar.monthrange(year, month)[1]
        month_end_date_object = datetime(year, month, final_day_in_month)
        # filter for specific country and dates
        covid_data_country_timeframe = filter_dataframe_country_timeframe(covid_data, country_name, month_start_date_object,
                                                                          month_end_date_object)
        for param in parameters_of_interest:
            single_param_df = covid_data_country_timeframe[[param, "stringency_index"]]
            single_param_df_filtered = single_param_df.dropna()
            file_path = os.path.join(raw_data_country_path, f"{country_name}_{param}_{month}_{year}_df_filtered.csv")
            single_param_df_filtered.to_csv(file_path)
            if single_param_df_filtered.size > 5:
                x = single_param_df_filtered[param]
                x_one_dimensional = x.values.reshape(-1, 1)  # must convert to 1D array for R^2 calculation
                y = single_param_df_filtered["stringency_index"]
                model = LinearRegression()
                print(f"Looking for R^2 for {country_name} between {start_date} and {end_date}, param {param}")
                print(f"Size of filtered df is {single_param_df_filtered.size}")
                model.fit(x_one_dimensional, y)
                r_squared = model.score(x_one_dimensional, y)
            else:
                r_squared = None
                print(f"Did not look for R^2 value for {country_name} between {start_date} and {end_date} for"
                      f"{param}, only {single_param_df_filtered.size - 1} datapoints available")
            df_row = pd.DataFrame.from_records([{"country": country_name,
                                                 "range_start": month_start_date_object,
                                                 "range_end": month_end_date_object,
                                                 "indicator": param,
                                                 "correlation_to_stringency": r_squared}])
            df = pd.concat([df, df_row], axis=0)
    file_path = os.path.join(country_path, f"{country_name}_monthly_correlations.csv")
    df.to_csv(file_path)


def plot_graphs_whole_timerange() -> None:
    pass


def plot_scatter(independent_variable: str, df: pd.DataFrame, country_name: str, country_path: Union[Path, str],
                 start_date: str, end_date: str, dependent_variable: str = "stringency_index") -> None:
    """
    Given an independent variable (such as 'new_cases'), and a dataframe, produce a scatter plot and
    save it as an image. Image file name is defined on penultimate line of this function.
    
    :param independent_variable: str name of variable whose effect we are trying to quantify
    :param df: dataframe containing data to be plotted
    :param country_name: str country name (to label plot)
    :param country_path: path to country's folder to save results in
    :param start_date: str begin of date range (to label plot)
    :param end_date: str end of date range (to label plot)
    :param dependent_variable: name of the dependent variable, defaults to 'stringency_index'
    :return: None
    """
    if independent_variable == dependent_variable:
        log.info(f"independent and dependent variable are both {independent_variable}, aborting")
        return
    msg = f"Plotting scatter withOUT trendline, country is {country_name} independent variable is {independent_variable}" \
          f" and dependent variable is {dependent_variable}"
    log.info(msg)
    # create folder to save results in
    folder_path = os.path.join(country_path, "graphs", "scatter_no_trendline")
    os.makedirs(folder_path, exist_ok=True)
    # only get the two columns we are interested in from dataframe and drop NA values
    df = df[[dependent_variable, independent_variable]].dropna()
    if df.size < 5:
        msg = "Fewer than 5 records, no point in scatter"
        print(msg)
        log.info(msg)
        return
    x = df[independent_variable]
    y = df[dependent_variable]
    plt.scatter(x=x, y=y)
    # add title and labels for axes
    plt.suptitle(f"{country_name} between {start_date} and {end_date}")
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    # save as image
    filepath = os.path.join(folder_path, f"scatter_{dependent_variable}_{independent_variable}_{country_name}.png")
    plt.savefig(filepath)
    plt.close()


def plot_scatter_with_trendline(independent_variable: str, df: pd.DataFrame, country_name: str,
                                country_path: Union[str, Path],
                                start_date: str, end_date: str, dependent_variable: str = "stringency_index") -> None:
    """
    Given an independent variable (such as 'new_cases'), and a dataframe, produce a scatter plot, including
    a trendline, its equation and an R^2 value. Image file name is defined on the penultimate line of this function.

    :param independent_variable: str name of variable whose effect we are trying to quantify
    :param df: dataframe containing data to be plotted
    :param country_name: str country name (to label plot)
    :param start_date: str begin of date range (to label plot)
    :param end_date: str end of date range (to label plot)
    :param dependent_variable: name of the dependent variable, defaults to 'stringency_index'
    :return: None
    """
    if independent_variable == dependent_variable:
        log.info(f"independent and dependent variable are both {independent_variable}, aborting")
        return
    msg = f"Plotting scatter with trendline, country is {country_name} independent variable is {independent_variable}" \
          f" and dependent variable is {dependent_variable}"
    log.info(msg)
    # create folder to save results in
    folder_path = os.path.join(country_path, "graphs", "scatter_trendline")
    os.makedirs(folder_path, exist_ok=True)
    # only get the two columns we are interested in from dataframe and drop NA values
    df = df[[dependent_variable, independent_variable]].dropna()
    if df.size < 5:
        msg = "Fewer than 5 records, no point in scatter"
        print(msg)
        log.info(msg)
        return
    x = df[independent_variable]
    y = df[dependent_variable]
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
    file_path = os.path.join(folder_path,
                             f"scatter_trendline_{dependent_variable}_{independent_variable}_{country_name}.png")
    plt.savefig(file_path)
    plt.close()


def multiple_linear_regression(predictors: List[str], df: pd.DataFrame, country_name: str,
                               country_path: Union[str, Path]) -> None:
    """
    Given a list of predictors (independent variables) for which we have data in the given data frame, produce
    a multiple linear regression model based on all the given predictors.

    :param predictors: list of str - column names of predictors of 'stringency_index'
    :param df: dataframe containing the data
    :param country_name: str country name
    :return: None
    """
    folder_path = os.path.join(country_path, "multiple_linear_regression")
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print("Folder already exists, continuing")
    interested_columns = predictors.copy()
    interested_columns.append("stringency_index")
    # filter only columns we are interested in
    df = df[interested_columns]
    # drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()
    file_path = os.path.join(folder_path, f"df_mutliple_regression_data_{country_name}.csv")
    df.to_csv(file_path)
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
                                             start_date: str, end_date: str, country_path: Union[str, Path]) -> None:
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
    folder_path = os.path.join(country_path, "multiple_linear_regression")
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print("Folder already exists, continuing")
    interested_columns = predictors.copy()
    interested_columns.append("stringency_index")
    # filter only columns we are interested in
    df = df[interested_columns]
    # drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()
    file_path = os.path.join(folder_path, f"df_multiple_regression_data_{country_name}_alt.csv")
    df.to_csv(file_path)
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
    file_path = os.path.join(folder_path, f"predicted_vs_actual_values_alt_{country_name}.csv")
    pred_df.to_csv(file_path)
    pred_df = pred_df[["actual_value", "predicted_value"]]
    pred_df = pred_df.dropna()
    plot_scatter("actual_value", pred_df, country_name, country_path, start_date, end_date, "predicted_value")
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
    file_path = os.path.join(folder_path, f"correlation_coefficients_multiple_regression_alt_{country_name}.csv")
    coeff_df.to_csv(file_path)
    print(f"ALT: Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"ALT: Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"ALT: Root Mean Squared Error: {sqrt(mean_squared_error(y_test, y_pred))}")
