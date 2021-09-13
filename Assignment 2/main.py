########################################################################################################################
# EX1
########################################################################################################################


'''
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...

(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters

(2) find duplicates and exclude them
    - remove only the unnecessary rows

(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutff value for idnetifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers

(4) impute missing values using the KNNImputer
    - including the excluded outliers!
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!

(5) find the class distribution
    - use the groupby() method

(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
'''

########################################################################################################################
# Solution
########################################################################################################################
import numpy
import pandas as pd
from util import delimiter_fix
from scipy.stats import norm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # set pandas options to make sure you see all info when printing dfs
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    missing_types = [numpy.NAN, 'missing', -999]

    # load file and correct delimiters ... delimiter_fix() fixes all faulty lines but one (off by 1 entry)
    file = delimiter_fix('data/wine_exercise.csv')

    # load data (drop bad lines due to imperfect delimiter fix)
    data = pd.read_csv(file, delimiter=';', error_bad_lines=False, skiprows=1, skipfooter=1)

    # consistently encode NA values (to numpy NaN)
    data = data.replace(missing_types, numpy.NAN)

    # fix and encode values in col 'season' (SPRING => 0, etc.)
    data = data.replace(
        to_replace=[r'(?i)(spr)', r'(?i)(sum)', r'(?i)(aut)', r'(?i)(win)'],
        value=[0, 1, 2, 3],
        regex=True)

    # split country-age into respective cols
    data = data.rename(columns={'country-age': 'country'})
    data['age'] = 0
    for i, val in enumerate(data.iloc[:, -2].values.tolist()):
        split = str(val).split('-')
        data.iloc[i, -2] = split[0]         # country
        data.iloc[i, -1] = split[1][0]      # age (remove 'years')

    # remove duplicate rows
    data = data.drop_duplicates()

    # remove outliers
    # proline
    data.loc[:, 'proline'] = data.loc[:, 'proline'].replace(to_replace=0, value=numpy.NAN)

    # fix magnesium and total_phenols in row 165 by hand to be able to continue
    data.iloc[165, 4] = 111
    data.iloc[165, 5] = 1.7

    # magnesium
    magnesium_vals = [int(string) for string in data.loc[:, 'magnesium']]
    norm(magnesium_vals)
    mean = norm.mean()
    print(mean)
    pdf = norm.pdf(magnesium_vals, mean, 1)
    print(pdf)

    # print(data)