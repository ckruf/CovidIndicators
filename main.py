from typing import Optional
from app_logging import log
from download_dataset import download
from analyze_data import analyze_data


DATASET_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

# PARAMETERS - change these to modify behaviour

COUNTRIES = ["Sweden", "United Kingdom"]
# date range we are interested in, start date will be floored to month and end date will be ceilinged to month
# (so if start date is 2020-03-10, the program will automatically start at beginning of March 2020 and if
# end is 2020-12-20 ut will automatically end at end of December 2020)
START_DATE = "2020-03-01"
END_DATE = "2020-12-31"
# Here, you can manually set parameters you are interested in
PARAMETERS_OF_INTEREST = ["icu_patients", "hosp_patients", "total_cases", "total_deaths", "new_deaths", "new_cases"]
# You can optionally specify a path to the target folder/directory in which you want the results to be saved.
# In this folder, the program will create a folder called 'Results' with appropriate subfolders.
# If None is provided, the Results folder will be created in this project.
RESULTS_FOLDER: Optional[str] = None


def main() -> None:
    log.info("Running...")
    file_path = download(DATASET_URL)
    analyze_data(filepath_covid_data=file_path,
                 countries_list=COUNTRIES,
                 start_date=START_DATE,
                 end_date=END_DATE,
                 parameters_of_interest=PARAMETERS_OF_INTEREST,
                 target_folder=RESULTS_FOLDER)


if __name__ == '__main__':
    main()
