from app_logging import log
from download_dataset import download
from analyze_data import analyze_data


DATASET_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

COUNTRY = "United Kingdom"
START_DATE = "2020-03-01"
END_DATE = "2020-12-31"


def main() -> None:
    log.info("Running...")
    file_path = download(DATASET_URL)
    analyze_data(file_path, COUNTRY, START_DATE, END_DATE)


if __name__ == '__main__':
    main()
