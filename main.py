from app_logging import log
from download_dataset import download
from analyze_data import analyze_data


DATASET_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

COUNTRY = "Sweden"
START_DATE = "2020-03-01"
END_DATE = "2020-12-31"
# Pearson correlation threshold to plot graph (ie if R > 0.45 or R < -0.45, graph will be plotted)
MIN_CORRELATION = 0.45
# if true will automatically plot scatters (without trendlines) for all variables
# who pass MIN_CORRELATION threshold
PLOT_ALL = False
# same as above, but trendline included, with equation and R^2 value
PLOT_ALL_TRENDLINE = True
# if True, will do multiple regression using first method (without graph of predicted vs actual)
MUTLIPLE_REGRESSION = True
# if True, will doe
MULTIPLE_REGRESSION_ALT_SCATTER = True
# if True drop 'per_million' and 'per_thousand' data, because within same country, correlation to
# per capita version of variable is same as to absolute number of variable (ie we will only get 'icu_patients',
# not 'icu_patients', 'icu_patients_per_thousand', 'icu_patients_per_million'
DROP_PER = True



def main() -> None:
    log.info("Running...")
    file_path = download(DATASET_URL)
    analyze_data(file_path, COUNTRY, START_DATE, END_DATE, MIN_CORRELATION, PLOT_ALL, PLOT_ALL_TRENDLINE,
                 MUTLIPLE_REGRESSION, MULTIPLE_REGRESSION_ALT_SCATTER)


if __name__ == '__main__':
    main()
