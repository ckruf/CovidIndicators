from app_logging import log
from download_dataset import download
from analyze_data import analyze_data


DATASET_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

COUNTRY = "Sweden"
START_DATE = "2020-03-01"
END_DATE = "2020-12-31"
MIN_CORRELATION = 0.45
PLOT_ALL = False
PLOT_ALL_TRENDLINE = True
MUTLIPLE_REGRESSION = True
MULTIPLE_REGRESSION_ALT_SCATTER = True



def main() -> None:
    log.info("Running...")
    file_path = download(DATASET_URL)
    analyze_data(file_path, COUNTRY, START_DATE, END_DATE, MIN_CORRELATION, PLOT_ALL, PLOT_ALL_TRENDLINE,
                 MUTLIPLE_REGRESSION, MULTIPLE_REGRESSION_ALT_SCATTER)


if __name__ == '__main__':
    main()
