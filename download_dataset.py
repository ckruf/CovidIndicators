import os
import requests
from typing import Optional
from app_logging import log


def download(url: str, dest_folder: Optional[str] = None, redownload: bool = False) -> str:
    """
    Download a file from the given url and save it in the given folder. If no folder is given,
    the file is downloaded to the folder in which this script is located. If the given destination folder does not
    exist, ValueError is raised. If the file already exists, then it will not be downloaded again, unless
    redownload = True. The filename of the downloaded file is parsed from the URL. The function returns
    the path to the downloaded file.

    :param url: URL from which the file should downloaded
    :param dest_folder: path of destination folder to which the file should be saved
    :param redownload: bool should file be redownloaded if it already exists?
    :return: path to downloaded file
    """
    if dest_folder is None:
        dest_folder = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(dest_folder):
        raise ValueError("The given destination folder does not exist.")

    filename = url.split("/")[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)

    if os.path.exists(file_path):
        if not redownload:
            log.info("File already exists, no download necessary")
            return file_path

    log.info(f"Downloading file from {url} to {file_path}")

    r = requests.get(url, stream=True)
    if r.ok:
        log.info(f"saving to {os.path.abspath(file_path)}")
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return file_path
    else:
        msg = f"Download failed: status code - {r.status_code}, \n request text - {r.text}"
        log.error(msg)
        raise ConnectionError(msg)


def delete_files() -> None:
    """
    Delete all .csv (except owid_covid_data.csv) and .png files in this directory. To be used between runs.

    :return: None
    """
    pass
