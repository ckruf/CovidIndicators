from app_logging import log
from download_dataset import download
from analyze_data import analyze_data


DATASET_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"


def main() -> None:
    log.info("Running...")
    file_path = download(DATASET_URL)
    analyze_data(file_path, "Czech Republic")


if __name__ == '__main__':
    main()
