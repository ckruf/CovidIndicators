import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn import linear_model


def analyze_data(filepath: str, country_name: str) -> None:
    """
    Given the filepath to the covid data csv file, run linear regressions.

    :param filepath: path to the csv file
    :return: None
    """
    covid_data = pd.read_csv(filepath)
    print(covid_data.head())
    covid_data_country = covid_data.query(f"location == '{country_name}'")
    print(covid_data_country.head())
    if covid_data_country.shape[0] < 1:
        raise ValueError("No data exists for given country")
    stringency_corr_mat = covid_data_country.corr("pearson")
    sorted_mat = stringency_corr_mat.unstack().sort_values()["stringency_index"]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(sorted_mat)


