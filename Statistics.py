import pandas as pd
import matplotlib.pyplot as plt

file_path = "csv/data/filename.csv"

df = pd.read_csv(file_path)

def difference_pd(df):
    data = df.iloc[:, 1:]
    diff_df = data.diff()
    diff_df.insert(0,'Step', df.iloc[:,0])
    return diff_df

diff_df = difference_pd(df)

steps_column = df.iloc[:, 0]
photon_columns = df.iloc[:, 15:21]
photon_diff_columns = diff_df.iloc[:, 15:21]

# Create subplots for individual columns and their differences
num_columns = photon_columns.shape[1]
fig, axs = plt.subplots(num_columns, 1, figsize=(10, 6 * num_columns))

for i, ax in enumerate(axs):
    ax.plot(steps_column, photon_columns.iloc[:, i], label= photon_columns.columns[i])
    ax.plot(steps_column, photon_diff_columns.iloc[:, i], label= photon_diff_columns.columns[i]+' diff')
    ax.set_ylabel(photon_columns.columns[i])
    ax.legend()

plt.xlabel('Step')
plt.show()



output_file_path = "csv/data/output.csv"
diff_df.to_csv(output_file_path,index=False)