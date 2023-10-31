# flake8: noqa
import logging
import refinitiv.data as rd
import os
import pandas as pd
# %cd /workspaces/Strategy/

def open_session() -> None:
    """Uses the local config file to open a session with the Refinitiv Data Platform. This makes sure the path is only defined once, so if the file moves, we only have to change it here.
    """
    os.environ["RD_LIB_CONFIG_PATH"] = "/workspaces/Strategy/Research/CODE/config/"
    rd.open_session()

def get_dir_path(folder: str = "data") -> str:
    return f"/workspaces/Strategy/Research/CODE/{folder}/"

def download_data(
    ric: str,
    frequency: str = "daily",
    start: str = "1900-01-01",
    end: str = "2100-01-01",
    fields: list = None,
    folder: str = "data",
    force_redownload: bool = False,
    check_errors: bool = True,
) -> pd.DataFrame:
    """Dowloads data from Refinitiv Data Platform and saves it to a csv file in the data/category folder. If the data already exists it will be loaded from the csv file.
    If the dataset has been commited to the repository, it is save to call this function without a preceeding call to open_session(), since the local csv file will be used.

    Args:
        ric (str): Identifies the data to be downloaded
        frequency (str): frequency of the data to be downloaded
        start (str, optional): first date of the data. Defaults to "1900-01-01".
        end (str, optional): last date of the date. Defaults to "2100-01-01".
        fields (list, optional): list of fields to be downloaded. If None all available Fields are requestet. Defaults to None.
        folder (str, optional): name of the folder the data will be saved to. Defaults to "data".
        force_redownload (bool, optional): if True the data will be downloaded even if it already exists. Defaults to False.
        check_errors (bool, optional): if True the script will throw an error if the data is empty or if one column of the dataframe contains only NaN values. Defaults to True.
        
    Returns:
        pd.DataFrame: Dataframe containing the requested data
    """
    dir_path = get_dir_path(folder)
    new_ric = ric.split("/")[0]
    file_path = f"{dir_path}{new_ric}.csv"
    

    if os.path.exists(file_path) and not force_redownload:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    assert "RD_LIB_CONFIG_PATH" in os.environ, f"If the data is not commited to this repository, you need to call open_session() before you can download data. File {file_path} does not exist."

    # create folder where the data will be saved
    os.makedirs(dir_path, exist_ok=True)

    df = rd.get_history(
                universe=ric,
                # interval=frequency,
                start=start,
                end=end,
                fields=fields,
            )

    # save data to csv file
    try:
        df.to_csv(file_path)
    except UnboundLocalError:
        print("No data could be assigned before trying to save the df")

    if check_errors:
        # throw an error if the df is empty
        if df.empty:
            raise ValueError(
                "Refinitiv Data Platform returned an empty dataframe. This is likey to invalid combination of ric, frequency, start and end date."
            )
        # Throw an error if one column df contains only NaN values
        if (fields is not None) and (df.isna().all(axis=0).any()):
            raise ValueError(
                f"Refinitiv Data Platform returned an empty column for field {df[df.isna().all(axis=0)]}. This Field is likely not available for the ric {ric}"
            )

    logging.info(
        f"Data of ric: {ric} was successfully safed under {file_path} ({len(df)} rows have been downloaded)"
    )

    return df
