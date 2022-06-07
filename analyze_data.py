import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
from typing import Optional, List
from math import sqrt
import os

from app_logging import log


def correlation_matrix(df: pd.DataFrame, description: Optional[str] = None,
                       correlation_param: str = "stringency_index", min_correlation: float = 0.45) -> List[str]:
    """
    Given a pandas dataframe, a correlation variable and a description, print a sorted correlation matrix for how all
    variables correlate to the given correlation parameter (by default 'stringency_index'). Saves the produced
    correlation matrix as a .csv and returns a list of parameters whose (absolute) Pearson R value is greater than the
    min_correlation value (defaults to 0.45).

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
    sorted_mat.to_csv(f"full_correlation_matrix.csv")
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


def analyze_data(filepath: str,
                 single_country: bool,
                 country_name: str,
                 countries_list: List[str],
                 parameters_of_interest: List[str],
                 compare_directly: bool = False,
                 start_date: str = "2020-01-01",
                 end_date: str = "2022-03-31",
                 auto_corr_parameters: bool = True,
                 min_correlation: float = 0.45,
                 scatter_plot: bool = False,
                 scatter_plot_trendline: bool = False,
                 multiple_regression: bool = False,
                 multiple_regression_alt_trendline: bool = False,
                 drop_per: bool = False,
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

    :param filepath: path to the owid_covid_data.csv file
    :param country_name: name of country we are interested in
    :param start_date: str in ISO format representing start of time range we are interested in
    :param end_date: str in ISO format representing end of time range we are interested in
    :param min_correlation:
    :param scatter_plot: if True, scatter plots for all variables with correlation greater than min_correlation
    :param scatter_plot_trendline: if True, scatter plots w/ trendline for all variables with correlation > min_correlation
    :param multiple_regresion: if True, produce multiple linear regression model using first method, w/o scatter
    :param multiple_regression_alt_trendline: if True, produce multiple regression model using second method, w/ scatter
    :param drop_per: if True, drop _per_thousand and _per_million data from being plotted (since corr. same as absolute)
    :param target_folder: path to folder in which to create Results folder.
    :return: None
    """
    if target_folder is None:
        # get current working directory
        pwd = os.path.dirname(os.path.realpath(__file__))
        try:
            path = os.path.join(pwd, "Results")
            os.mkdir(path)
        except FileExistsError:
            print("The Results directory already exists, continuing")
        except FileNotFoundError:
            print("The provided path does not exist, please provide correct path")
            return

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
        raise ValueError("No data exists for given country - either misspelled or non-existent.")
    # show number of data points for country for all time
    print_df_size(covid_data_country, country_name)
    # save correlation matrix for all time
    _: List[str] = correlation_matrix(covid_data_country, f"{country_name} all time corr mat")

    # convert start date and end date into datetime objects so we can use them to filter
    start_date_object: datetime = datetime.fromisoformat(start_date)
    end_date_object: datetime = datetime.fromisoformat(end_date)

    # time range query
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] > start_date_object) &
                                                      (covid_data_country['date'] < end_date_object)]

    # show number of data points for given time range
    print_df_size(covid_data_country_timeframe, f"{country_name} from {start_date} to {end_date}")
    # save correlation matrices (full version and version filtered for min_correlation)
    correlated = correlation_matrix(df=covid_data_country_timeframe,
                                    description=f"{country_name} {start_date} to {end_date} corr mat",
                                    min_correlation=min_correlation)
    # filter correlated params to remove 'per_thousand' and 'per_million' params, because within
    # same country, absolute quantity is enough
    if drop_per:
        correlated = list(filter(lambda x: "per_thousand" not in x and "per_million" not in x, correlated))

    # make scatters for all parameters greater than min_correlation
    if scatter_plot:
        for param in correlated:
            plot_scatter(param, covid_data_country_timeframe, country_name, start_date, end_date)
    if scatter_plot_trendline:
        for param in correlated:
            plot_scatter_with_trendline(param, covid_data_country_timeframe, country_name, start_date, end_date)
    if multiple_regression:
        multiple_linear_regression(correlated, covid_data_country_timeframe, country_name, start_date, end_date)
    if multiple_regression_alt_trendline:
        alt_multiple_linear_regression_with_plot(correlated, covid_data_country_timeframe, country_name, start_date,
                                                 end_date)


def analyze_data_single_country(filepath: str,
                                country_name: str,
                                parameters_of_interest: List[str],
                                start_date: str = "2020-01-01",
                                end_date: str = "2022-03-31",
                                auto_corr_parameters: bool = True,
                                min_correlation: float = 0.45,
                                scatter_plot: bool = False,
                                scatter_plot_trendline: bool = False,
                                multiple_regression: bool = False,
                                multiple_regression_alt_trendline: bool = False,
                                drop_per: bool = False,
                                target_folder: Optional[str] = None) -> None:
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
        raise ValueError("No data exists for given country - either misspelled or non-existent.")
    # show number of data points for country for all time
    print_df_size(covid_data_country, country_name)
    # save correlation matrix for all time
    _: List[str] = correlation_matrix(covid_data_country, f"{country_name} all time corr mat")

    # convert start date and end date into datetime objects so we can use them to filter
    start_date_object: datetime = datetime.fromisoformat(start_date)
    end_date_object: datetime = datetime.fromisoformat(end_date)

    # time range query
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] > start_date_object) &
                                                      (covid_data_country['date'] < end_date_object)]
    # save correlation matrices in time range (full version and version filtered for min_correlation)
    correlated = correlation_matrix(df=covid_data_country_timeframe,
                                    description=f"{country_name} {start_date} to {end_date} corr mat",
                                    min_correlation=min_correlation)
    if auto_corr_parameters:
        parameters = correlated
        if drop_per:
            parameters = list(filter(lambda x: "per_thousand" not in x and "per_million" not in x, parameters))
    else:
        parameters = parameters_of_interest

    # make scatters for all parameters of interest (if option selected)
    if scatter_plot:
        for param in parameters:
            plot_scatter(param, covid_data_country_timeframe, country_name, start_date, end_date)
    if scatter_plot_trendline:
        for param in parameters:
            plot_scatter_with_trendline(param, covid_data_country_timeframe, country_name, start_date, end_date)
    if multiple_regression:
        multiple_linear_regression(parameters, covid_data_country_timeframe, country_name, start_date, end_date)
    if multiple_regression_alt_trendline:
        alt_multiple_linear_regression_with_plot(parameters, covid_data_country_timeframe, country_name, start_date,
                                                 end_date)


def plot_scatter(independent_variable: str, df: pd.DataFrame, country_name: str,
                 start_date: str, end_date: str, dependent_variable: str = "stringency_index") -> None:
    """
    Given an independent variable (such as 'new_cases'), and a dataframe, produce a scatter plot and
    save it as an image. Image file name is defined on penultimate line of this function.
    
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
    msg = f"Plotting scatter withOUT trendline, country is {country_name} independent variable is {independent_variable}" \
          f" and dependent variable is {dependent_variable}"
    log.info(msg)
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
    plt.savefig(f"scatter_{dependent_variable}_{independent_variable}_{country_name}.png")
    plt.close()


def plot_scatter_with_trendline(independent_variable: str, df: pd.DataFrame, country_name: str,
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
    plt.savefig(f"scatter_trendline_stringency_index_{independent_variable}_{country_name}.png")
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
    df = df.dropna()
    df.to_csv(f"df_multiple_regression_data_{country_name}.csv")
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
    df = df.dropna()
    df.to_csv(f"df_multiple_regression_data_{country_name}_alt.csv")
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
    pred_df.to_csv(f"predicted_vs_actual_values_alt_{country_name}.csv")
    pred_df = pred_df[["actual_value", "predicted_value"]]
    pred_df = pred_df.dropna()
    plot_scatter("actual_value", pred_df, country_name, start_date, end_date, "predicted_value")
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
    coeff_df.to_csv(f"correlation_coefficients_multiple_regression_alt_{country_name}.csv")
    print(f"ALT: Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"ALT: Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"ALT: Root Mean Squared Error: {sqrt(mean_squared_error(y_test, y_pred))}")
