#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import re
from tensorflow.keras.models import Model
import streamlit as st
from PIL import Image
import numpy as np
from numpy import  vstack
import zipfile
import io
import pandas as pd
import torch 
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import altair as alt
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')

st.set_page_config(
    page_title="Twitter Sentimental Analysis",
    page_icon="✅",
    layout="wide",
)
df = pd.read_csv('tweets100kfinalk.csv')
model = torch.load('model50000.pt',map_location=torch.device('cpu'))
model.eval()
def run_sentiment_analysis(text):
    # Assuming you have already trained the model and have the trained model object

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        # Tokenize and encode the input text
    input_encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            padding=True,
            max_length=128,  # Adjust the maximum sequence length as per your training data
            return_tensors='pt'  # Return PyTorch tensors
    )
    input_ids = input_encoding['input_ids'].to(device)  # Move tensors to the same device as the model
    attention_mask = input_encoding['attention_mask'].to(device)  # Move tensors to the same device as the model
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    _, predicted_label = torch.max(probabilities, dim=1)
    
    # Determine sentiment label and color
    if predicted_label == 0:
        sentiment_label = "Negative"
        sentiment_color = "red"
        sentiment_emoji = "😞"
    else:
        sentiment_label = "Positive"
        sentiment_color = "#33FF57"  # Bright green color
        sentiment_emoji = "😃"

    return sentiment_label, sentiment_color, sentiment_emoji

# Display the title and text input area
st.title("Twitter Sentimental Analysis")
txt = st.text_area('Enter Tweet to Analyze', '')

# Display sentiment analysis result with color and emoji using Markdown
if st.button('Analyze Sentiment'):
    sentiment_label, sentiment_color, sentiment_emoji = run_sentiment_analysis(txt)
    st.write('', f'<font color="{sentiment_color}" size="+10">{sentiment_label} {sentiment_emoji}</font>', unsafe_allow_html=True)

    if sentiment_label=='Positive':
        st.snow()

st.divider()



st.header("Sentiment Timeline")

# custom_color_scheme = alt.Scale(scheme='set1').copy()
custom_color_scheme = alt.Scale(
    range=['#e41a1c', '#00FF00', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
)
timeline = grouped_data = df.groupby('date').agg({
    'N_prob': ['mean', 'count'],  # Calculate the mean and count of 'N_prob' for each day
    'P_prob':'mean' # Calculate the mean of 'P_prob' for each day
}).reset_index()
timeline.columns = ['Date', 'Negative', 'Total_count', 'Positive']
timeline = pd.melt(timeline,id_vars=['Date', 'Total_count'], value_vars=['Negative', 'Positive'], 
                      var_name='Sentiment', value_name='Probability')
# st.dataframe(timeline)
def create_line_chart(data):
    chart = alt.Chart(data).mark_line().encode(
        x='Date:T',
        y='Probability:Q',
        color=alt.Color('Sentiment:N', scale=custom_color_scheme),  # Specify the color scheme here
         tooltip=[alt.Tooltip('date:T', title='Day of Week'),
             alt.Tooltip('Probability:Q', title='Percentage'),
             alt.Tooltip('Total_count:Q', title='Total_count')]
    ).properties(
        title=''
    )
    return chart
chart = create_line_chart(timeline)

st.altair_chart(chart,theme="streamlit", use_container_width=True)

st.divider()
st.header("Analysis on Weekdays")

grouped_data = df.groupby('DayofWeek').agg({
    'N_prob': 'mean',  # Calculate the mean and count of 'N_prob' for each day
    'P_prob': ['mean', 'count']   # Calculate the mean and count of 'P_prob' for each day
}).reset_index()

# Flatten the multi-level column index
grouped_data.columns = ['DayofWeek', 'Negative', 'Positive', 'Total_count']

# Display the updated grouped data
order_of_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
grouped_data['DayofWeek'] = pd.Categorical(grouped_data['DayofWeek'], categories=order_of_weekdays, ordered=True)
grouped_data = grouped_data.sort_values('DayofWeek')
melted_data = pd.melt(grouped_data, id_vars=['DayofWeek','Total_count'], value_vars=['Negative','Positive'])

# Create a grouped bar chart using Altair

chart = alt.Chart(melted_data).mark_bar().encode(
    column=alt.Column('DayofWeek', spacing=40,title=""),
    x=alt.X('variable',title="",axis=alt.Axis(labels=False)),
    y=alt.Y('value:Q',title="Percentage"),
    color=alt.Color('variable', scale=custom_color_scheme),  # Specify the color scheme here
         tooltip=[alt.Tooltip('DayofWeek', title='Day of Week'),
             alt.Tooltip('value:Q', title='Percentage'),
             alt.Tooltip('Total_count:Q', title='Total_count')]
).configure_view(stroke='transparent').configure_legend(
    title=None,   # Remove the legend title
    orient='right'  # Set the legend position to bottom
)

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.dataframe(grouped_data)


# Melt the DataFrame to have a 'variable' column

# chart = chart.properties(
#     title=alt.TitleParams(text='Negative and Positive Comparison', anchor='middle', align='center')
# )

with fig_col2:
    st.altair_chart(chart,theme="streamlit")
    st.write(" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Monday&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Tuesday &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;      Wednesday&nbsp;&nbsp; Thrusday&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Friday&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;Saturday&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Sunday")
df['date'] = pd.to_datetime(df['date'], format='%B %d %Y').dt.date
st.divider()
st.header("WordCloud on Weekdays")

# Sort the DataFrame by 'date'
df = df.sort_values('date')
df = df.reset_index()
del(df['index'])
import re
def tweet_cleaner(x):
    text=re.sub("[@&][A-Za-z0-9_]+","", x)     # Remove mentions
    text=re.sub(r"http\S+","", text)           # Remove media links
    return  pd.Series([text])
df[['plain_text']] = df.text.apply(tweet_cleaner)
#Convert all text to lowercase
df.plain_text = df.plain_text.str.lower()
#Remove newline character
df.plain_text = df.plain_text.str.replace('\n', '')
#Replacing any empty strings with null
df = df.replace(r'^\s*$', np.nan, regex=True)
if df.isnull().sum().plain_text == 0:
    print ('no empty strings')
else:
    df.dropna(inplace=True)

dayfilter = st.selectbox("Select the Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])   
from nltk.corpus import stopwords
col1, col2 = st.columns(2)

stop_words = set(stopwords.words('english'))

df1 = df[df['DayofWeek']==dayfilter]
df2 = df1[df1['Predicted Label']==0]

query_words={'recession', '#','http' }
stop_words.update(query_words)
for word in query_words:
    df2.plain_text = df2.plain_text.str.replace(word, '')
#Creating word cloud
from wordcloud import WordCloud, ImageColorGenerator
wc=WordCloud(stopwords=stop_words, collocations=False, max_font_size=55, max_words=50, background_color="black")
wc.generate(' '.join(df2.plain_text))
with col1:
    st.image(wc.to_array(),width=620)


#     st.write(df2)
df3 = df1[df1['Predicted Label']==1]

query_words={'recession', '#','http' }
stop_words.update(query_words)
for word in query_words:
    df3.plain_text = df3.plain_text.str.replace(word, '')
#Creating word cloud
from wordcloud import WordCloud, ImageColorGenerator
wc=WordCloud(stopwords=stop_words, collocations=False, max_font_size=55, max_words=50, background_color="black")
wc.generate(' '.join(df3.plain_text))

with col2:
# Display the word cloud using Streamlit
    st.image(wc.to_array(), width=620)
    
st.divider()
st.header("Top 10 Users")

col1,col2=st.columns(2)

# Group the data by user and calculate the mean and count of N_prob and P_prob
grouped_data = df.groupby('user').agg({
    'N_prob': ['mean', 'count'],  
    'P_prob': ['mean', 'count']   
}).reset_index()

# Flatten the multi-level column names
grouped_data.columns = ['user', 'N_prob_mean', 'N_count', 'P_prob_mean', 'P_count']

# Filter users with a count greater than 5
grouped_data = grouped_data[(grouped_data['N_count'] > 5) & (grouped_data['P_count'] > 5)]

# Sort the DataFrame based on N_prob_mean and N_prob_count in descending order
top_negative_users_df = grouped_data.sort_values(['N_prob_mean', 'N_count'], ascending=[False, False]).head(10)

# Sort the DataFrame based on P_prob_mean and P_prob_count in descending order
top_positive_users_df = grouped_data.sort_values(['P_prob_mean', 'P_count'], ascending=[False, False]).head(10)




with col1:


    # Create a pie chart using altair
    chart = alt.Chart(top_negative_users_df).mark_arc().encode(
        theta='N_count:Q',
        color='user:N',
        tooltip=['user:N', 'N_count:Q']
    ).properties(
        title='Top Ten Users with Most Negative Tweets'
    )
    # Display the pie chart
    st.altair_chart(chart, use_container_width=True)

with col2:


    # Create a pie chart using altair
    chart = alt.Chart(top_positive_users_df).mark_arc().encode(
        theta='P_count:Q',
        color='user:N',
        tooltip=['user:N', 'P_count:Q']
    ).properties(
        title='Top Ten Users with Most Positive Tweets'
    )
    # Display the pie chart
    st.altair_chart(chart, use_container_width=True)

# st.line_chart(timeline.set_index('date'), use_container_width=True)
#st.line_chart(data=timeline,x=timeline.date,y=timeline.Probability)
