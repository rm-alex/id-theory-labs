import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_data(data):
    for col in ['gas', 'hot_water', 'central_heating']:
        data[col] = data[col].map({'Yes': 1, 'No': 0})
    data = pd.get_dummies(
        data,
        columns=['extra_area_type_name'],
        drop_first=True
    )
    data = pd.get_dummies(
        data,
        columns=['district_name'],
        drop_first=True
    )
    return data.drop(columns=['index'])

def heatmap(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(16, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='crest', fmt='.2f')
    plt.title('Heatmap')
    plt.show()

df=pd.read_csv('lab3/Archive2025/data.csv')
df=process_data(df)

df['floor_ratio']=df['floor']/df['floor_max']

heatmap(df)

to_drop = ['gas', 'hot_water', 'ceil_height', 
           'extra_area_count', 'floor', 'bath_count', 
           'other_area', 'floor_max', 'year', 
           'extra_area_type_name_loggia']

df = df.drop(columns=to_drop)

heatmap(df)
print(df.columns)

df.to_csv('lab3/processed_data.csv', index=False)

df_test = pd.read_csv('lab3/Archive2025/test.csv')
df_test = process_data(df_test)
df_test['floor_ratio']=df_test['floor']/df_test['floor_max']
df_test = df_test.drop(columns=to_drop)
df_test.to_csv('lab3/processed_test_data.csv', index=False)