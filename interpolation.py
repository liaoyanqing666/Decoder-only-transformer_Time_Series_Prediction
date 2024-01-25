import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# Filter some unreasonable data
def filter(input_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train.csv',
           output_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_filter.csv',
           k=None):
    """
    :param k: The maximum number of non-numeric values allowed
    """
    df = pd.read_csv(input_file, index_col=0, header=0)
    if k is None:
        k = df.shape[1]

    filtered_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Filtering Rows"):
        if row.dropna().size >= k:
            filtered_rows.append(index)

    filtered_df = df.loc[filtered_rows]
    filtered_df.to_csv(output_file)

# cubic spline interpolation
def cubic_spline(input_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_filter.csv',
                 output_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_cubic.csv'):
    df = pd.read_csv(input_file, index_col=0, header=0)
    df.columns = pd.to_datetime(df.columns, format='%Y-%m-%d')

    def interpolate_nan_values(row):
        not_nan_indices = row[~row.isna()].index

        # Create a cubic spline function using non-NaN indices and values
        cs = CubicSpline(not_nan_indices, row[not_nan_indices], bc_type='natural')

        # Find NaN indices
        nan_indices = row[row.isna()].index

        # Interpolate NaN values
        row.loc[nan_indices] = cs(nan_indices)

        return row

    tqdm.pandas(desc="Interpolating Rows")
    df_interpolated = df.progress_apply(interpolate_nan_values, axis=1)

    df_interpolated.to_csv(output_file)

# nearest neighbor interpolation
def nearest_neighbor(input_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_filter.csv',
                     output_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_nearest.csv'):
    df = pd.read_csv(input_file, index_col=0, header=0)
    df.columns = pd.to_datetime(df.columns, format='%Y-%m-%d')

    # Use nearest neighbor interpolation along columns
    df_interpolated = df.interpolate(method='nearest', axis=1)

    df_interpolated.to_csv(output_file)

# Median fill that distinguishes between weekends and weekdays
def weekday(input_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_filter.csv',
            output_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_weekday.csv'):
    df = pd.read_csv(input_file, index_col=0, header=0)
    df.columns = pd.to_datetime(df.columns, format='%Y-%m-%d')

    def custom_interpolate(row):
        weekend_indices = row.index.weekday >= 5  # 5和6分别代表周六和周日

        weekend_median = None
        weekday_median = None
        if np.any(np.logical_and(weekend_indices, ~np.isnan(row))):
            weekend_median = row[weekend_indices].median()
        if np.any(np.logical_and(~weekend_indices, ~np.isnan(row))):
            weekday_median = row[~weekend_indices].median()

        row.loc[row.isna() & ~weekend_indices] = (weekday_median if weekday_median is not None else weekend_median)
        row.loc[row.isna() & weekend_indices] = (weekend_median if weekend_median is not None else weekday_median)
        # print(weekday_median, weekend_median)

        return row

    tqdm.pandas(desc="Weekday&weekend Interpolating Rows")
    df_custom_interpolated = df.progress_apply(custom_interpolate, axis=1)

    df_custom_interpolated.to_csv(output_file)

# 0 padding
def zero_padding(input_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_filter.csv',
                 output_file='E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测\\train_zero_padding.csv'):
    df = pd.read_csv(input_file, index_col=0, header=0)
    df.columns = pd.to_datetime(df.columns, format='%Y-%m-%d')

    zero_padded_df = df.copy()

    for column in df.columns:
        zero_padded_df[column].fillna(0, inplace=True)

    zero_padded_df.to_csv(output_file)

if __name__ == '__main__':
    # filter(k=80)

    # cubic_spline()
    # nearest_neighbor()
    # zero_padding()
    weekday()
