


import pandas as pd

df = pd.read_csv('tweets1.csv',names=['target','id','date','flag','user','text'], encoding='latin-1')
df = df.drop(['id','flag'],axis=1)
n = 100000  # Replace this with the number of rows you want to select
df = df.sample(n)

# If you want to reset the index of the sampled rows, use reset_index with drop=True
df = df.reset_index(drop=True)
from datetime import datetime

def process_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%a %b %d %H:%M:%S PDT %Y')

    # Extract the required components
    day_of_week = date_obj.strftime('%A')
    month = date_obj.strftime('%B')
    day = date_obj.strftime('%d')
    year = date_obj.strftime('%Y')

    return day_of_week, month+" "+day+" "+year


# Apply the process_date function to the date_column and create new columns
df[['DayofWeek','date']] = df['date'].apply(process_date).apply(pd.Series)
df['target'] = df['target'].replace(4, 1)

# Drop the original date_column if needed

df.to_csv('tweets100kfinalf.csv', index=False)



