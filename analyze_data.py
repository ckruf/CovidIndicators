import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
from typing import Optional, List, Union
from math import sqrt
import os
from pathlib import Path

from app_logging import log


def filter_dataframe_country_timeframe(df: pd.DataFrame, country: str, start_date: Optional[str] = None,
                                       end_date: Optional[str] = None) -> pd.DataFrame:
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
    print_df_size(covid_data_country, f"{country} all time")
    if start_date is None or end_date is None:
        return covid_data_country
    # convert start date and end date into datetime objects so we can use them to filter
    start_date_object: datetime = datetime.fromisoformat(start_date)
    end_date_object: datetime = datetime.fromisoformat(end_date)
    # time range query
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] > start_date_object) &
                                                      (covid_data_country['date'] < end_date_object)]
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

    :param filepath_covid_data: path to the owid_covid_data.csv file
    :param country_name: name of country we are interested in
    :param start_date: str in ISO format representing start of time range we are interested in
    :param end_date: str in ISO format representing end of time range we are interested in
    :param min_correlation:
    :param scatter_plot: if True, scatter plots for all variables with correlation greater than min_correlation
    :param scatter_plot_trendline: if True, scatter plots w/ trendline for all variables with correlation > min_correlation
    :param multiple_regression: if True, produce multiple linear regression model using first method, w/o scatter
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

    if auto_corr_parameters:
        if single_country:
            analyze_data_single_country(
                covid_data=covid_data,
                country_name=country_name,
                parameters_of_interest=parameters_of_interest,
                start_date=start_date,
                end_date=end_date,
                auto_corr_parameters=auto_corr_parameters,
                min_correlation=min_correlation,
                scatter_plot=scatter_plot,
                scatter_plot_trendline=scatter_plot_trendline,
                multiple_regression=multiple_regression,
                multiple_regression_alt_trendline=multiple_regression_alt_trendline,
                drop_per=drop_per,
                target_folder_path=target_folder_path
            )
        else:
            if compare_directly:
                parameters_of_interest = []
                for country in countries_list:
                    filtered_df = filter_dataframe_country_timeframe(
                        df=covid_data,
                        country=country,
                        start_date=start_date,
                        end_date=end_date
                    )
                    correlated_params = correlation_matrix(
                        df=filtered_df,
                        description=f"{country} between {start_date} - {end_date}",
                        min_correlation=min_correlation,
                        save=False
                    )
                    parameters_of_interest.extend(correlated_params)
                for country in countries_list:
                    analyze_data_single_country(
                        covid_data=covid_data,
                        country_name=country,
                        parameters_of_interest=parameters_of_interest,
                        start_date=start_date,
                        end_date=end_date,
                        auto_corr_parameters=False,
                        min_correlation=None,
                        scatter_plot=scatter_plot,
                        scatter_plot_trendline=scatter_plot_trendline,
                        multiple_regression=multiple_regression,
                        multiple_regression_alt_trendline=multiple_regression_alt_trendline,
                        drop_per=drop_per,
                        target_folder_path=target_folder_path
                    )
            else:
                for country in countries_list:
                    analyze_data_single_country(
                        covid_data=covid_data,
                        country_name=country,
                        parameters_of_interest=parameters_of_interest,
                        start_date=start_date,
                        end_date=end_date,
                        auto_corr_parameters=auto_corr_parameters,
                        min_correlation=min_correlation,
                        scatter_plot=scatter_plot,
                        scatter_plot_trendline=scatter_plot_trendline,
                        multiple_regression=multiple_regression,
                        multiple_regression_alt_trendline=multiple_regression_alt_trendline,
                        drop_per=drop_per,
                        target_folder_path=target_folder_path
                    )
    else:
        if single_country:
            analyze_data_single_country(
                covid_data=covid_data,
                country_name=country_name,
                parameters_of_interest=parameters_of_interest,
                start_date=start_date,
                end_date=end_date,
                auto_corr_parameters=auto_corr_parameters,
                min_correlation=min_correlation,
                scatter_plot=scatter_plot,
                scatter_plot_trendline=scatter_plot_trendline,
                multiple_regression=multiple_regression,
                multiple_regression_alt_trendline=multiple_regression_alt_trendline,
                drop_per=drop_per,
                target_folder_path=target_folder_path
            )
        else:
            for country in countries_list:
                analyze_data_single_country(
                    covid_data=covid_data,
                    country_name=country,
                    parameters_of_interest=parameters_of_interest,
                    start_date=start_date,
                    end_date=end_date,
                    auto_corr_parameters=auto_corr_parameters,
                    min_correlation=min_correlation,
                    scatter_plot=scatter_plot,
                    scatter_plot_trendline=scatter_plot_trendline,
                    multiple_regression=multiple_regression,
                    multiple_regression_alt_trendline=multiple_regression_alt_trendline,
                    drop_per=drop_per,
                    target_folder_path=target_folder_path
                )


def analyze_data_single_country(covid_data: pd.DataFrame,
                                country_name: str,
                                parameters_of_interest: Optional[List[str]] = None,
                                start_date: str = "2020-01-01",
                                end_date: str = "2022-03-31",
                                auto_corr_parameters: bool = True,
                                min_correlation: Optional[float] = 0.45,
                                scatter_plot: bool = False,
                                scatter_plot_trendline: bool = False,
                                multiple_regression: bool = False,
                                multiple_regression_alt_trendline: bool = False,
                                drop_per: bool = False,
                                target_folder_path: Optional[Path] = None) -> Optional[List[str]]:

    # create folder for country
    country_path = os.path.join(target_folder_path, country_name)
    try:
        os.mkdir(country_path)
    except FileExistsError:
        print("Country folder already exists, continuing")

    # show sample of dataframe and print all columns
    print(covid_data.head())
    column_list = covid_data.columns.values.tolist()
    print(f"All columns: {column_list}")
    print_df_size(covid_data, "all countries")

    # filter for specific country and dates
    covid_data_country_timeframe = filter_dataframe_country_timeframe(covid_data, country_name, start_date, end_date)
    # save correlation matrices in time range (full version and version filtered for min_correlation)
    correlated = correlation_matrix(df=covid_data_country_timeframe,
                                    description=f"{country_name} {start_date} to {end_date}",
                                    min_correlation=min_correlation if auto_corr_parameters else 0,
                                    country_path=country_path)
    if auto_corr_parameters:
        parameters = correlated
        if drop_per:
            parameters = list(filter(lambda x: "per_thousand" not in x and "per_million" not in x, parameters))
    else:
        parameters = parameters_of_interest

    # make scatters for all parameters of interest (if option selected)
    if scatter_plot:
        for param in parameters:
            plot_scatter(param, covid_data_country_timeframe, country_name, country_path, start_date, end_date)
    if scatter_plot_trendline:
        for param in parameters:
            plot_scatter_with_trendline(param, covid_data_country_timeframe, country_name, country_path, start_date, end_date)
    if multiple_regression:
        multiple_linear_regression(parameters, covid_data_country_timeframe, country_name, country_path)
    if multiple_regression_alt_trendline:
        alt_multiple_linear_regression_with_plot(parameters, covid_data_country_timeframe, country_name, start_date,
                                                 end_date, country_path)
    if auto_corr_parameters:
        return parameters


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


def plot_scatter_with_trendline(independent_variable: str, df: pd.DataFrame, country_name: str, country_path: Union[str, Path],
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
