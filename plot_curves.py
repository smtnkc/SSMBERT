
import glob
import pandas as pd
import matplotlib.pyplot as plt

file_pattern = 'logs/heterogrouped*.csv'
file_list = glob.glob(file_pattern)

# load and concatenate
dfs = [pd.read_csv(f) for f in file_list]
combined = pd.concat(dfs, ignore_index=True)

# compute average per epoch
avg = combined.groupby('Epoch')[['Train_Loss', 'Val_Loss']].mean().reset_index()

# plotting
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()

ax.plot(avg['Epoch'], avg['Train_Loss'], label='Train Loss')
ax.plot(avg['Epoch'], avg['Val_Loss'], label='Validation Loss')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Average Train and Validation Loss per Epoch')
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend()
ax.grid(True)

# save to PDF
plt.savefig('loss_curves.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)
