import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
import calendar
from typing import Optional, List, Tuple
import os
from pathlib import Path
# import all necessary third party libraries

# import logging tool
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
    # print dataframe size after filtering for country
    print_df_size(covid_data_country, f"{country} all time has size")
    # if no start and end dates were provided, return the dataframe filtered for country only
    if start_date is None or end_date is None:
        return covid_data_country
    # time range query betwen given dates
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] >= start_date) &
                                                      (covid_data_country['date'] <= end_date)]
    # if filtered dataframe has less than 1 record, raise error
    if covid_data_country_timeframe.size < 1:
        raise ValueError(f"No data exists for {country} between dates {start_date} and {end_date}")
    # print size of dataframe filtered for country and date range
    print_df_size(covid_data_country_timeframe, f"{country} between {start_date} and {end_date}")

    # return filtered data frame
    return covid_data_country_timeframe


def print_df_size(df: pd.DataFrame, description: Optional[str] = None) -> None:
    """
    Given a pandas dataframe and a description, print its size (number of records).

    :param df: dataframe whose size we want to print
    :param description: string description of dataframe to be printed
    :return: None
    """
    msg = f"\nDataframe {description} has size {df.size}\n"
    print(msg)
    log.info(msg)


def analyze_data(filepath_covid_data: str,
                 countries_list: List[str],
                 parameters_of_interest: List[str],
                 start_date: str = "2020-01-01",
                 end_date: str = "2022-03-31",
                 target_folder: Optional[str] = None,
                 predictive_repetitions: int = 10) -> None:
    """
    Find correlations between given indicators and stringency index over time, save the data as .csv files and
    plot and save scatter plots showing the data.

    :param filepath_covid_data: path to the owid_covid_data.csv file
    :param countries_list: list of country names which we are interested in
    :param parameters_of_interest: list of porameters we are interested in
    :param start_date: str in ISO format representing start of time range we are interested in
    :param end_date: str in ISO format representing end of time range we are interested in
    :param target_folder: path to folder in which to create Results folder.
    :param predictive_repetitions: number of R^2 values to obtain using predictive model
    :return: None
    """
    log.debug(f"Running analysis, filepath={filepath_covid_data}, countries={countries_list}, "
              f"parameters={parameters_of_interest}, target_folder={target_folder}")
    # if no target path to store results is provided, we will create Results folder in the location where
    # this script is located
    if target_folder is None:
        # get current working directory
        pwd = os.path.dirname(os.path.realpath(__file__))
        try:
            # try to create Results directory to store results
            path = os.path.join(pwd, "Results")
            target_folder_path = Path(path)
            os.mkdir(path)
        # if folder already exists, continue the program
        except FileExistsError:
            msg = "The Results directory already exists, continuing"
            print(msg)
            log.info(msg)
    # target location was provided, create Results folder there
    else:
        # attempt to create folder
        try:
            target_folder_path: Path = Path(target_folder)
            os.mkdir(target_folder_path)
        # if provided path is not valid, raise exception
        except Exception as e:
            msg = "The provided path is not valid"
            print(msg)
            log.error(msg)
            raise e
    log.debug(f"target folder path is {target_folder_path}")

    # read full dataframe with data for all countries, in full time range
    covid_data = pd.read_csv(filepath_covid_data)
    # convert 'date' column to datetime format, so we can filter by date later
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # create list to store all summary dataframes created for each country containing correlations for
    # each month for all countries, to be able to use the dataframes' data for plotting graphs
    countries_dfs: List[Tuple[pd.DataFrame, str]] = []

    # iterate over all countries in countries list to find all monthly correlations for all indicators for each country
    for country in countries_list:
        country_df = find_monthly_correlations(covid_data=covid_data,
                                               country_name=country,
                                               parameters_of_interest=parameters_of_interest,
                                               start_date=start_date,
                                               end_date=end_date,
                                               target_folder_path=target_folder_path,
                                               repetitions=predictive_repetitions)
        countries_dfs.append((country_df, country))

    # iterate over sumarry dataframes for each country to use the data in them to plot graphs
    for df, country_name in countries_dfs:
        # plot graphs for country
        plot_graphs_country_whole_timerange(
            df=df,
            country=country_name,
            parameters_of_interest=parameters_of_interest,
            results_folder=target_folder_path
        )


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
    # create datetime object from timestamp of start date
    start = datetime(start_year, start_month, 1)
    # create datetime object from timestamp of end date
    end = datetime(end_year, end_month, 1)
    # return a list of tuples containing all months (including year) between the start month and end month
    return [(d.month, d.year) for d in rrule(MONTHLY, dtstart=start, until=end)]


def find_monthly_correlations(covid_data: pd.DataFrame,
                              country_name: str,
                              parameters_of_interest: List[str],
                              start_date: str,
                              end_date: str,
                              target_folder_path: Path,
                              repetitions: int) -> pd.DataFrame:
    """
    Find correlations between parameters of interest and stringency index for one country for each whole month
    between start date and end date. Put data in pandas dataframe and return it.

    :param covid_data: pandas dataframe containing all covid data
    :param country_name: name of country we are interested in
    :param parameters_of_interest: list of str parameters we are interested in
    :param start_date: beginning of month
    :param end_date: end of month
    :param target_folder_path: folder to save results in
    :param repetitions: number of R^2 values to obtain using predictive model
    :return:
    """
    log.info(f"Running analysis for {country_name}")
    # create folder for country, to store results
    country_path = os.path.join(target_folder_path, country_name)
    try:
        os.mkdir(country_path)
    except FileExistsError:
        msg = "Country folder already exists, continuing"
        print(msg)
        log.info(msg)

    # create folder for raw data, to store results
    raw_data_country_path = os.path.join(country_path, "raw_data")
    try:
        os.mkdir(raw_data_country_path)
    except FileExistsError:
        msg = "Raw data folder for country already exists, continuing"
        print(msg)
        log.debug(msg)

    # create folder for unfiltered raw data (before N/A values are dropped) to store results
    raw_data_country_unfiltered_path = os.path.join(raw_data_country_path, "unfiltered")
    try:
        os.mkdir(raw_data_country_unfiltered_path)
    except FileExistsError:
        msg = "Unfiltered folder for country already exists, continuing"
        print(msg)
        log.debug(msg)

    # show sample of dataframe and print all columns
    print(covid_data.head())
    # show all columns in data set
    column_list = covid_data.columns.values.tolist()
    msg = f"All columns: {column_list}"
    print(msg)
    log.info(msg)
    # show size of data set
    print_df_size(covid_data, "all countries")

    # convert begin date and end date of range into datetime objects
    start_date_object = datetime.fromisoformat(start_date)
    end_date_object = datetime.fromisoformat(end_date)
    # find all the months in the given time range
    months = find_months(start_date_object.month, start_date_object.year, end_date_object.month, end_date_object.year)
    log.debug(f"months={months}")

    # define columns for data that will be stored
    fixed_columns = ["country",
                     "range_start",
                     "range_end",
                     "time_label",
                     "no_of_values",
                     "stringency_range",
                     "stringency_month_average",
                     "indicator",
                     "indicator_month_average",
                     "correlation_to_stringency",
                     "alt_correlation_to_stringency",
                     "adj_correlation_to_stringency",
                     ]

    # create dataframe to store data obtained in analysis
    df = pd.DataFrame(columns=fixed_columns)

    # iterate over each month to get data for that month
    for month, year in months:
        # create datetime object representing beginning of month
        month_start_date_object = datetime(year, month, 1)
        # find final day of the given month (30th vs 31st)
        final_day_in_month = calendar.monthrange(year, month)[1]
        # create datetime object representing end of month
        month_end_date_object = datetime(year, month, final_day_in_month, 23, 59, 59)
        # filter  for specific country and dates (between start and end of given month)
        covid_data_country_timeframe = filter_dataframe_country_timeframe(covid_data,
                                                                          country_name,
                                                                          month_start_date_object,
                                                                          month_end_date_object)
        # iterate over all indicators we are interested in (for each month)
        for param in parameters_of_interest:
            # filter dataframe with containing all data (many indicators) for the given country
            # to only include one indicator we are interested in, stringency index and the date the indicator was measured
            single_param_df = covid_data_country_timeframe[[param, "stringency_index", "date"]]
            # create file path where to save .csv file containing the filtered data set
            file_path = os.path.join(raw_data_country_unfiltered_path,
                                     f"{country_name}_{param}_{month}_{year}_df_unfiltered.csv")
            # save the .csv file
            single_param_df.to_csv(file_path)
            # drop N/A values
            single_param_df_filtered = single_param_df.dropna()
            # sort values by value of the indicator, so trends are more obvious in visual inspection
            single_param_df_filtered.sort_values(by=param, ascending=True)
            # create file path for dataset with dropped N/A values
            file_path_filtered = os.path.join(
                raw_data_country_path, f"{country_name}_{param}_{month}_{year}_df_filtered.csv")
            # save N/A filtered dataframe into .csv file
            single_param_df_filtered.to_csv(file_path_filtered)
            # find the range in stringency index values in the given month, because when there was no change
            # in stringency index for the whole month, there is no point in finding correlation
            stringency_range = single_param_df_filtered["stringency_index"].max() - single_param_df_filtered[
                "stringency_index"].min()
            # find average value of stringency index for the month, to see trends
            stringency_month_avg = single_param_df_filtered["stringency_index"].mean()
            # find average value of the given indicator for the month, to see trends
            indicator_month_avg = single_param_df_filtered[param].mean()
            msg = f"Stringency range for {country_name} between {month_start_date_object} and {month_end_date_object}" \
                  f" is {stringency_range}"
            print(msg)
            log.info(msg)

            # initialize values to None in case there is no attempt at calculating correlation
            r_squared = None
            r_squared_alt = None
            r_squared_adj = None

            # if we have less than five values of the indicator in the month, correlation will not be calculated,
            # because more values are needed for a reasonable accurate calculation
            if single_param_df_filtered.shape[0] < 5:
                msg = f"Did not look for R^2 value for {country_name} between {month_start_date_object} " \
                      f"and {month_end_date_object} for {param}, only {single_param_df_filtered.shape[0]} " \
                      f"data points available"
                print(msg)
                log.info(msg)

            # if the range in stringency index is less than 0.2 on a 100 point scale, there is no point in trying
            # to find correlation, since stringency index did not change enough
            elif stringency_range < 0.2:
                msg = "Did not look for R^2 values due to insufficient change in stringency index"
                log.info(msg)
            # if both of the two above conditons are false, we can calculate the correlation
            else:
                # get our x and y values (the values for the indicator and for the stringency index)
                x = single_param_df_filtered[param]
                y = single_param_df_filtered["stringency_index"]
                x_one_dimensional = x.values.reshape(-1, 1)  # must convert to 1D array for R^2 calculation
                # create LinearRegression object from sklearn library as it will allow us to calculate R^2
                model = LinearRegression()
                msg = f"Looking for R^2 for {country_name} between {month_start_date_object} and " \
                      f"{month_end_date_object}, param {param}"
                print(msg)
                log.info(msg)
                msg = f"Number of rows of df is {single_param_df_filtered.shape[0]}"
                print(msg)
                log.info(msg)
                # fit x and y into linear model
                model.fit(x_one_dimensional, y)
                # use sklearn function score to find R^2 values
                r_squared = model.score(x_one_dimensional, y)
                # using statsmodels.api OLS function to find R^2 values, to check consistency
                result = sm.OLS(y, sm.add_constant(x)).fit()
                # statsmodels.api regular R^2
                r_squared_alt = result.rsquared
                # statsmodels.api adjusted R^2
                r_squared_adj = result.rsquared_adj

            # collect all relevant data for summary spreadsheet into one dictionary
            fixed_results = {"country": country_name,
                             "range_start": month_start_date_object,
                             "range_end": month_end_date_object,
                             "no_of_values": single_param_df_filtered.shape[0],
                             "stringency_range": stringency_range,
                             "stringency_month_average": stringency_month_avg,
                             "indicator": param,
                             "indicator_month_average": indicator_month_avg,
                             "correlation_to_stringency": r_squared,
                             "alt_correlation_to_stringency": r_squared_alt,
                             "adj_correlation_to_stringency": r_squared_adj
                             }

            row = fixed_results
            # write row of summary data into summary data dataframe
            df_row = pd.DataFrame.from_records([row])
            df = pd.concat([df, df_row], axis=0)

    # get text names for months (March 2020 instead of 03-2020)
    df["time_label"] = df["range_start"].apply(lambda z: z.strftime("%B %Y"))
    # save summary dataframe into .csv file
    file_path = os.path.join(country_path, f"{country_name}_monthly_correlations.csv")
    df.to_csv(file_path)
    # return dataframe with summary data, as it will be used to plot graphs
    return df


def plot_graphs_country_whole_timerange(df: pd.DataFrame, country: str, parameters_of_interest: List[str],
                                        results_folder: Path) -> None:
    """
    Given a dataframe containing the correlations between parameters of interest and stringency index over time,
    produce time series scatter plots of the evolution of correlation over time.

    :param df: pandas dataframe containing correlation to stringency over time
    :param country: str name of country
    :param parameters_of_interest: list of parameters for which we want to know correlations to stringency index
    :param results_folder: folder containing results and subfolders for countr
    :return: None, only saves graphs as .png files.
    """
    # create graphs folder to save graphs to
    graphs_folder = os.path.join(results_folder, country, "graphs")
    try:
        os.mkdir(graphs_folder)
    # if folder already exists, continue the program
    except FileExistsError:
        print(f"Graphs folder for {country} already exists")
    # iterate over all indicators to create a graph for each one by one
    for param in parameters_of_interest:
        # filter the summary dataframe for a single indicator only, since it contains all indicators
        single_param_df = df.query(f"indicator == '{param}'")
        # get x and y values from dataframe
        y = single_param_df["correlation_to_stringency"]
        x = single_param_df["range_start"]
        # plot scatter graph
        plt.scatter(x=x, y=y)
        # add main title
        plt.suptitle(f"{country}")
        # add sub title
        plt.title(f"R^2 between {param} and stringency index")
        # label x axis
        plt.xlabel("Time")
        # label y axis
        plt.ylabel("R^2")
        # rotate x axis labels 90 degrees, so that they don't overlap
        plt.xticks(rotation=90)
        # this function is necessary so labels fit on graph
        plt.tight_layout()
        # create path for image file into which graph will be stored
        image_path = os.path.join(graphs_folder, f"{param}_stringency_{country}.png")
        # save image with graph
        plt.savefig(image_path)
        # reset matplotlib plot, otherwise graphs can get mixed up
        plt.close()
