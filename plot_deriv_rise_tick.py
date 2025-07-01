'''
Plotting DERIV_MAX vs RISE_TIME and DERIV_MAX vs TICK_SHIFT
This script reads event df, preprocesses it, and plots the relationship between DERIV_MAX and RISE_TIME, DERIV_MAX and TICK_SHIFT.
'''

import os
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii


# Set the matplotlib parameters for the plot
plt.rcParams['font.size'] = 10
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["legend.loc"] = "best"
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.fancybox"] = False
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = False


def table_to_pandas(file, hdu_number):
    """
    Convert an Astropy Table from a FITS file to a Pandas dfFrame.
    Args:
        file (str): Path to the FITS file.
        hdu_number (int): HDU number to read from the FITS file.
    Returns:
        pd.dfFrame: dfFrame containing the table df.
    """
    tbl = Table.read(file, hdu=hdu_number)
    names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    df = tbl[names].to_pandas()
    return df


def merge_fits_pandas(event_files, hdu_number):
    """
    Merge multiple FITS files into a single Pandas dfFrame.
    Args:
        event_files (list): List of paths to FITS files.
        hdu_number (int): HDU number to read from the FITS files.
    Returns:
        pd.dfFrame: Merged dfFrame containing df from all files.
    """
    i = 0
    for event_file in tqdm(event_files):
        if i == 0:
            df = table_to_pandas(event_file, hdu_number)
            i += 1
        elif i == 1:
            _df = table_to_pandas(event_file, hdu_number)
            df = pd.concat([df, _df], ignore_index=True)
    return df


def preprocess(df):
    """
    Preprocess the event dfFrame.
    Filters the df based on specific conditions and adds a random shift to the TICK_SHIFT column.
    Args:
        df (pd.dfFrame): Input dfFrame containing event data.
    Returns:
        pd.dfFrame: Preprocessed dfFrame with filtered data and modified TICK_SHIFT.
    """
    # Filter the event grade
    df = df[df["ITYPE"] <= 4]
    # Filter the event pixel
    df = df[df["PIXEL"] != 12]
    # Filter the event PHA
    df = df[df["PHA"] > 0]
    # Filter the event RISE_TIME
    df = df[df["RISE_TIME"] > 0]
    df = df[df["RISE_TIME"] < 128]
    # Filter the event DERIV_MAX
    df = df[df["DERIV_MAX"] > 0]
    # Randomly shift the TICK_SHIFT column to visualize the distribution
    upper_limit = 0.5
    lower_limit = -0.5
    random_values = np.random.uniform(lower_limit, upper_limit, size=(len(df),))
    df.loc[:, "TICK_SHIFT"] += random_values
    return df


def linear_function(x, a, b):
    """
    Linear function for plotting.
    """
    return a*x + b


def step_function_upper(x):
    """
    Step function for upper screening line.
    Args:
        x (np.ndarray): Input array of DERIV_MAX values.
    Returns:
        np.ndarray: Output array with step function values.
    """

    # Initialize the result array with zeros
    result = np.zeros_like(x)

    # Define the step function conditions
    result[x < 100] = -4
    result[(x >= 100) & (x < 200)] = -3
    result[(x >= 200) & (x < 300)] = -1
    result[(x >= 300) & (x < 400)] = 0
    result[(x >= 400) & (x < 500)] = 1
    result[(x >= 500) & (x < 1000)] = 2
    result[(x >= 1000) & (x < 2000)] = 3
    result[x >= 2000] = 4
    return result


def step_function_below(x):
    """
    Step function for below screening line.
    Args:
        x (np.ndarray): Input array of DERIV_MAX values.
    Returns:
        np.ndarray: Output array with step function values.
    """

    # Initialize the result array with zeros
    result = np.zeros_like(x)

    # Define the step function conditions
    result[x < 100] = -8
    result[(x >= 100) & (x < 200)] = -7
    result[(x >= 200) & (x < 300)] = -5
    result[(x >= 300) & (x < 400)] = -4
    result[(x >= 400) & (x < 500)] = -3
    result[(x >= 500) & (x < 1000)] = -2
    result[(x >= 1000) & (x < 2000)] = -1
    result[(x >= 2000) & (x < 6000)] = 0
    result[x >= 6000] = 1
    return result


def plot_deriv_rise(df):
    """
    Plot DERIV_MAX vs RISE_TIME distribution
    Args:
        df (pd.dfFrame): Input dfFrame containing DERIV_MAX and RISE_TIME
    """
    # Set up the plot
    fig, ax = plt.subplots()
    # Set the fraction of the df to sample
    frac_value = 1
    df = df.sample(frac=frac_value)
    # Scatter plot of DERIV_MAX vs RISE_TIME
    grade_list = ["Hp", "Mp", "Ms"]
    for grade in range(0, len(grade_list)):
        df_grade = df[df["ITYPE"] == grade]
        ax.scatter(df_grade["DERIV_MAX"], df_grade["RISE_TIME"], s=1, color=cm.magma(grade/len(grade_list)), label=grade_list[grade])

    # Plot the screening lines
    z = np.linspace(min(df["DERIV_MAX"]), max(df["DERIV_MAX"]), 10000)
    params_upper = [-7.5e-4, 58]
    params_below = [params_upper[0], 46]
    ax.plot(z, linear_function(z, params_upper[0], params_upper[1]), color='red', label='upper')
    ax.plot(z, linear_function(z, params_below[0], params_below[1]), color='blue', label='below')

    plt.legend()
    plt.xlabel("DERIV_MAX")
    plt.ylabel("RISE_TIME")
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_deriv_tick(df):
    """
    Plot DERIV_MAX vs TICK_SHIFT distribution
    Args:
        df (pd.dfFrame): Input dfFrame containing DERIV_MAX and TICK_SHIFT
    """
    # Set up the plot
    fig, ax = plt.subplots()
    # Set the fraction of the df to sample
    frac_value = 1
    df = df.sample(frac=frac_value)
    # Scatter plot of DERIV_MAX vs RISE_TIME
    grade_list = ["Hp", "Mp", "Ms"]
    for grade in range(0, len(grade_list)):
        data_grade = df[df["ITYPE"] == grade]
        ax.scatter(data_grade["DERIV_MAX"], data_grade["TICK_SHIFT"], s=1, color=cm.magma(grade/len(grade_list)), label=grade_list[grade])

    # Plot the screening lines
    z = np.linspace(min(df["DERIV_MAX"]), max(df["DERIV_MAX"]), 10000)
    ax.plot(z, (step_function_upper(z)), color='red', label='upper')
    ax.plot(z, (step_function_below(z)), color='blue', label='below')

    plt.legend()
    plt.xlabel("DERIV_MAX")
    plt.ylabel("TICK_SHIFT")
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    """
    Main function to execute the script.
    Reads event df, preprocesses it, and plots the DERIV_MAX vs RISE_TIME.
    """

    # Define the observation ID and filter IDs
    OBSID = "300003010"
    FILTERID = ["0000", "1000", "5000"]
    for id in FILTERID:
        # Input directory and file pattern
        input_dir = "../doc/"
        input_file = f"xa{OBSID}rsl_p0px{id}_cl.evt.gz"
        input_event = os.path.join(input_dir, input_file)
        event_files = sorted(
            glob.glob(input_event)
        )

        # HDU number to read from the FITS files
        hdu_number = 1

        # Merge FITS files into a single dfFrame
        event_pandas = merge_fits_pandas(event_files, hdu_number)

        # Preprocess the df
        df = preprocess(event_pandas)

        # plot the df
        plot_deriv_rise(df)
        plot_deriv_tick(df)


if __name__ == "__main__":
    main()
