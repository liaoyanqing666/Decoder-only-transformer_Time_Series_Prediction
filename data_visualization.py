import os
import pandas as pd
import matplotlib.pyplot as plt

file_dir = 'E:\\大学本科课程\\大三2\\机器学习基础\\期末\\代码\\时序数据回归预测'
file_name = 'train_weekday.csv'

file = os.path.join(file_dir, file_name)
data_frame = pd.read_csv(file, header=0, index_col=0)

def plot_rows(data_frame, row_indices):
    plt.figure(figsize=(12, 6))

    for k in row_indices:
        row_data = data_frame.iloc[k]
        plt.plot(range(1, 551), row_data.values, label=f'Row {k}')

    plt.title('Change of Rows')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()


plot_rows(data_frame, [1, 2, 3, 4, 5])

