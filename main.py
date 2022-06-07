from typing import Optional
from app_logging import log
from download_dataset import download
from analyze_data import analyze_data


DATASET_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

# PARAMETERS - change these to modify behaviour

# choose whether the program should do analysis for one country or mutliple countries. You can still leave
# the COUNTRY and COUNTRIES parameters filled anyway, the program will use this switch ONLY to determine
# single country vs multi country behaviour
SINGLE_COUNTRY = True
# country whose data we are interested in
COUNTRY = "Sweden"
# list of countries we are interested in
COUNTRIES = ["Sweden", "United Kingdom"]
# If this is set to True, and AUTO_PARAMETERS is also true, then the program will find the most correlated parameters
# for each country, combine the parameters, and do analysis on the combined parameters for both countries. This setting
# will only make a difference if SINGLE_COUNTRY is False.
COMPARE_DIRECTLY = False
# date range we are interested in
START_DATE = "2020-03-01"
END_DATE = "2020-12-31"
# if this is True, then the program will look for parameters whose correlation to 'stringency_index' is greater
# than the MIN_CORRELATION and do analysis and plots for those parameters. If False, then PARAMETERS_OF_INTEREST
# will be used.
AUTO_CORR_PARAMETERS = True
# Pearson correlation threshold to plot graph (ie if R > 0.45 or R < -0.45, graph will be plotted)
MIN_CORRELATION = 0.45
# Here, you can manually set parameters you are interested in, for which
PARAMETERS_OF_INTEREST = ["icu_patients", "reproduction_rate", "positive_rate"]
# if true will automatically plot scatters (without trendlines) for all variables
# who pass MIN_CORRELATION threshold
SCATTER_PLOT = False
# same as above, but trendline included, with equation and R^2 value
SCATTER_PLOT_TRENDLINE = True
# if True, will do multiple regression using first method (without graph of predicted vs actual)
MUTLIPLE_REGRESSION = True
# if True, will do mutliple regression using secong method, with graph of predicted vs actual
MULTIPLE_REGRESSION_ALT_SCATTER = True
# if True drop 'per_million' and 'per_thousand' data, because within same country, correlation to
# per capita version of variable is same as to absolute number of variable (ie we will only get 'icu_patients',
# not 'icu_patients', 'icu_patients_per_thousand', 'icu_patients_per_million')
DROP_PER = True
# You can optionally specify a path to the target folder/directory in which you want the results to be saved.
# In this folder, the program will create a folder called 'Results' with appropriate subfolders.
# If None is provided, the Results folder will be created in this project.
RESULTS_FOLDER: Optional[str] = None


def main() -> None:
    log.info("Running...")
    file_path = download(DATASET_URL)
    analyze_data(filepath=file_path,
                 single_country=SINGLE_COUNTRY,
                 country_name=COUNTRY,
                 countries_list=COUNTRIES,
                 compare_directly=COMPARE_DIRECTLY,
                 start_date=START_DATE,
                 end_date=END_DATE,
                 auto_corr_parameters=AUTO_CORR_PARAMETERS,
                 min_correlation=MIN_CORRELATION,
                 parameters_of_interest=PARAMETERS_OF_INTEREST,
                 scatter_plot=SCATTER_PLOT,
                 scatter_plot_trendline=SCATTER_PLOT_TRENDLINE,
                 multiple_regression=MUTLIPLE_REGRESSION,
                 multiple_regression_alt_trendline=MULTIPLE_REGRESSION_ALT_SCATTER,
                 drop_per=DROP_PER,
                 target_folder=RESULTS_FOLDER)


if __name__ == '__main__':
    main()
