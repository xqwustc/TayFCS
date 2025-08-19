import pandas as pd
from sklearn.model_selection import train_test_split

data_path = './ipinyou_x2'

# Read the first 100 rows of train.csv
train_df = pd.read_csv(f'{data_path}/train_old.csv')

# Split the dataset into training and validation sets (7:1 ratio)
train_df_split, valid_df_split = train_test_split(train_df, test_size=1.0/7, random_state=2024)

# Save the datasets
train_df_split.to_csv(f'{data_path}/train.csv', index=False, encoding='utf-8')
valid_df_split.to_csv(f'{data_path}/valid.csv', index=False, encoding='utf-8')

# Output the split information
print('Train lines:', len(train_df_split))
print('Validation lines:', len(valid_df_split))
