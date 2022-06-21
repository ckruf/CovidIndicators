import numpy
import pandas as pd
import matplotlib.pyplot as plt
import pandas.core.series
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
import calendar
from typing import Optional, List, Tuple, Dict
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
    covid_data_country_timeframe = covid_data_country[(covid_data_country['date'] >= start_date) &
                                                      (covid_data_country['date'] <= end_date)]
    if covid_data_country_timeframe.size < 1:
        raise ValueError(f"No data exists for {country} between dates {start_date} and {end_date}")
    print_df_size(covid_data_country_timeframe, f"{country} between {start_date} and {end_date}")

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
    if target_folder is None:
        # get current working directory
        pwd = os.path.dirname(os.path.realpath(__file__))
        try:
            path = os.path.join(pwd, "Results")
            target_folder_path = Path(path)
            os.mkdir(path)
        except FileExistsError:
            msg = "The Results directory already exists, continuing"
            print(msg)
            log.info(msg)
        except FileNotFoundError:
            msg = "The provided path does not exist, please provide correct path"
            print(msg)
            log.info(msg)
            return
    else:
        try:
            target_folder_path: Path = Path(target_folder)
            os.mkdir(target_folder_path)
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

    countries_dfs: List[Tuple[pd.DataFrame, str]] = []

    for country in countries_list:
        country_df = find_monthly_correlations(covid_data=covid_data,
                                               country_name=country,
                                               parameters_of_interest=parameters_of_interest,
                                               start_date=start_date,
                                               end_date=end_date,
                                               target_folder_path=target_folder_path,
                                               repetitions=predictive_repetitions)
        countries_dfs.append((country_df, country))

    for df, country_name in countries_dfs:
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
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
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
    # create folder for country
    country_path = os.path.join(target_folder_path, country_name)
    try:
        os.mkdir(country_path)
    except FileExistsError:
        msg = "Country folder already exists, continuing"
        print(msg)
        log.info(msg)

    raw_data_country_path = os.path.join(country_path, "raw_data")
    try:
        os.mkdir(raw_data_country_path)
    except FileExistsError:
        msg = "Raw data folder for country already exists, continuing"
        print(msg)
        log.debug(msg)

    raw_data_country_unfiltered_path = os.path.join(raw_data_country_path, "unfiltered")
    try:
        os.mkdir(raw_data_country_unfiltered_path)
    except FileExistsError:
        msg = "Unfiltered folder for country already exists, continuing"
        print(msg)
        log.debug(msg)

    # show sample of dataframe and print all columns
    print(covid_data.head())
    column_list = covid_data.columns.values.tolist()
    msg = f"All columns: {column_list}"
    print(msg)
    log.info(msg)
    print_df_size(covid_data, "all countries")

    start_date_object = datetime.fromisoformat(start_date)
    end_date_object = datetime.fromisoformat(end_date)
    months = find_months(start_date_object.month, start_date_object.year, end_date_object.month, end_date_object.year)
    log.debug(f"months={months}")

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
                     "avg_pred_correlation_to_stringency"]

    predicted_columns = [f"pred_correlation_to_stringency_{i}" for i in range(repetitions)]

    df = pd.DataFrame(columns=fixed_columns)
    df_predictive_data = pd.DataFrame(columns=predicted_columns)

    for month, year in months:
        month_start_date_object = datetime(year, month, 1)
        final_day_in_month = calendar.monthrange(year, month)[1]
        month_end_date_object = datetime(year, month, final_day_in_month, 23, 59, 59)
        # filter for specific country and dates
        covid_data_country_timeframe = filter_dataframe_country_timeframe(covid_data,
                                                                          country_name,
                                                                          month_start_date_object,
                                                                          month_end_date_object)
        for param in parameters_of_interest:
            single_param_df = covid_data_country_timeframe[[param, "stringency_index", "date"]]
            file_path = os.path.join(raw_data_country_unfiltered_path,
                                     f"{country_name}_{param}_{month}_{year}_df_unfiltered.csv")
            single_param_df.to_csv(file_path)
            single_param_df_filtered = single_param_df.dropna()
            single_param_df_filtered.sort_values(by=param, ascending=True)
            file_path_filtered = os.path.join(raw_data_country_path, f"{country_name}_{param}_{month}_{year}_df_filtered.csv")
            single_param_df_filtered.to_csv(file_path_filtered)
            stringency_range = single_param_df_filtered["stringency_index"].max() - single_param_df_filtered[
                "stringency_index"].min()
            stringency_month_avg = single_param_df_filtered["stringency_index"].mean()
            indicator_month_avg = single_param_df_filtered[param].mean()
            msg = f"Stringency range for {country_name} between {month_start_date_object} and {month_end_date_object} " \
                  f"is {stringency_range}"
            print(msg)
            log.info(msg)

            r_squared = None
            r_squared_alt = None
            r_squared_adj = None
            predictive_r_squared_values = {}
            avg_predictive_r_squared = None

            if single_param_df_filtered.shape[0] < 5:
                msg = f"Did not look for R^2 value for {country_name} between {month_start_date_object} " \
                      f"and {month_end_date_object} for {param}, only {single_param_df_filtered.shape[0]} " \
                      f"data points available"
                print(msg)
                log.info(msg)

            elif stringency_range < 0.2:
                msg = "Did not look for R^2 values due to insufficient change in stringency index"
                log.info(msg)

            else:
                x = single_param_df_filtered[param]
                y = single_param_df_filtered["stringency_index"]
                x_one_dimensional = x.values.reshape(-1, 1)  # must convert to 1D array for R^2 calculation
                model = LinearRegression()
                msg = f"Looking for R^2 for {country_name} between {month_start_date_object} and " \
                      f"{month_end_date_object}, param {param}"
                print(msg)
                log.info(msg)
                msg = f"Number of rows of df is {single_param_df_filtered.shape[0]}"
                print(msg)
                log.info(msg)
                model.fit(x_one_dimensional, y)
                # using sklearn without prediction
                r_squared = model.score(x_one_dimensional, y)
                # using statsmodels.api OLS
                result = sm.OLS(y, sm.add_constant(x)).fit()
                # statsmodels.api regular R^2
                r_squared_alt = result.rsquared
                # statsmodels.api adjusted R^2
                r_squared_adj = result.rsquared_adj
                # R^2 values calculated using predictive model
                predictive_r_squared_values = calc_r_squared_predictive_model(x_one_dimensional, y, repetitions)
                avg_predictive_r_squared = calc_r_squared_predictive_model_avg(predictive_r_squared_values)

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
                             "adj_correlation_to_stringency": r_squared_adj,
                             "avg_pred_correlation_to_stringency": avg_predictive_r_squared}

            row = fixed_results

            df_row = pd.DataFrame.from_records([row])
            df = pd.concat([df, df_row], axis=0)

            df_predictive_row = pd.DataFrame.from_records([predictive_r_squared_values])
            df_predictive_data = pd.concat([df_predictive_data, df_predictive_row], axis=0)
    df["time_label"] = df["range_start"].apply(lambda z: z.strftime("%B %Y"))
    file_path = os.path.join(country_path, f"{country_name}_monthly_correlations.csv")
    df.to_csv(file_path)
    file_path_predictive = os.path.join(raw_data_country_path, f"{country_name}_predictive_correlation_data.csv")
    df_predictive_data.to_csv(file_path_predictive)
    return df


def calc_r_squared_predictive_model(x: numpy.ndarray, y: pandas.core.series.Series,
                                    repetitions: int) -> Dict[str, float]:
    """
    Given arrays with x and y values, calculate R^2 values using predictive model and return them in a dict
    of the form {'pred_correlation_to_stringency_<num>': <r^2 value>}

    :param x: numpy array containing x values
    :param y: pandas series containg y values (stringency index)
    :param repetitions: how many times to run predictions
    :return: dict with r^2 values, see docstring
    """
    results = {}
    for i in range(repetitions):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        pred_model = LinearRegression()
        pred_model.fit(x_train, y_train)
        y_pred = pred_model.predict(x_test)
        r_squared_pred = r2_score(y_test, y_pred)
        results[f"pred_correlation_to_stringency_{i}"] = r_squared_pred
    log.info(f"results={results}")
    return results


def calc_r_squared_predictive_model_avg(r_squared_values: Dict[str, float]) -> float:
    """
    Given a dictionary containing R^2 values, calculate the average R^2 value.

    :param r_squared_values: dictionary containing R^2 values as its values.
    :return: average R^2 values
    """
    total = 0
    for r_squared_value in r_squared_values.values():
        total += r_squared_value
    average = total / len(r_squared_values)
    log.info(f"average, calculated from {total} and {len(r_squared_values)} is {average}")
    return average


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
    graphs_folder = os.path.join(results_folder, country, "graphs")
    try:
        os.mkdir(graphs_folder)
    except FileExistsError:
        print(f"Graphs folder for {country} already exists")
    for param in parameters_of_interest:
        single_param_df = df.query(f"indicator == '{param}'")
        y = single_param_df["correlation_to_stringency"]
        x = single_param_df["range_start"]
        plt.scatter(x=x, y=y)
        plt.suptitle(f"{country}")
        plt.title(f"R^2 between {param} and stringency index")
        plt.xlabel("Time")
        plt.ylabel("R^2")
        image_path = os.path.join(graphs_folder, f"{param}_stringency_{country}.png")
        plt.savefig(image_path)
        plt.close()
