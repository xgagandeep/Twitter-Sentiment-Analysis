# Twitter Sentiment Analysis using BERT and TensorFlow

This repository provides a complete workflow for performing sentiment analysis on Twitter data using both PyTorch's BERT model and visualization tools like Streamlit. The project includes three core components:

1. **Data Preprocessing (`Preprocessing.py`)**: Cleans and processes the tweet dataset.
2. **Model Training (`Model_training.py`)**: Trains a BERT-based model for binary sentiment classification.
3. **Sentiment Analysis and Visualization (`model.py`)**: Performs real-time sentiment analysis using a pre-trained model and visualizes the results with interactive charts in a Streamlit web application.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Sentiment Analysis & Visualization](#sentiment-analysis--visualization)
- [Project Structure](#project-structure)

## Installation

To run this project, you need to install the required dependencies. First, clone this repository:

```bash
git clone https://github.com/your_username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for this project consists of tweets stored in `tweets1.csv`. Each tweet includes the following fields:

- `target`: Sentiment labels (0 for negative, 4 for positive; converted to 1 during preprocessing)
- `id`: Unique tweet ID
- `date`: Timestamp of the tweet
- `flag`: Quality flag
- `user`: Username of the tweet author
- `text`: The actual tweet content

After preprocessing, the dataset will contain the columns:
- `target`: Sentiment labels (0 for negative, 1 for positive)
- `user`: Username of the tweet author
- `text`: The actual tweet content
- `DayofWeek`: Day of the week when the tweet was posted
- `date`: Reformatted date of the tweet

## Data Preprocessing

The `Preprocessing.py` script handles data cleaning and preparation:

1. Load the dataset from `tweets1.csv`.
2. Drop unnecessary columns (`id`, `flag`).
3. Randomly sample 100,000 rows.
4. Extract date information (day of the week, month, and year).
5. Convert the positive sentiment label from `4` to `1`.

To run the script:

```bash
python Preprocessing.py
```

This will generate a processed dataset (`tweets100kfinalf.csv`).

## Model Training

The `Model_training.py` script trains a BERT-based sentiment classification model using the processed dataset:

1. Load the pre-trained BERT tokenizer and model.
2. Split the dataset into training (80%) and testing (20%) sets.
3. Tokenize the tweet text and encode it for BERT.
4. Train the model for 3 epochs.
5. Save the trained model to `model50000.pt`.

To run the model training:

```bash
python Model_training.py
```

The trained model will be saved as `model50000.pt`.

## Sentiment Analysis & Visualization

The `model.py` script runs the pre-trained sentiment analysis model and generates real-time interactive visualizations using Streamlit. It provides the following features:

- **Text Sentiment Analysis**: Enter a tweet and predict its sentiment (positive or negative).
- **Sentiment Timeline**: A line chart visualizing the trend of sentiments over time.
- **Analysis on Weekdays**: A bar chart displaying sentiment distributions across weekdays.
- **Word Cloud**: Displays word clouds of negative and positive tweets for the selected day of the week.
- **Top 10 Users**: Pie charts showing users with the highest number of negative and positive tweets.

To run the web app locally:

```bash
streamlit run model.py
```

The app will be accessible at `http://localhost:8501`.

## Project Structure

```
.
├── Preprocessing.py        # Data preprocessing script
├── Model_training.py       # Model training script
├── model.py                # Streamlit app for sentiment analysis and visualization
├── tweets1.csv             # Original dataset #too big of a file to upload on github
├── tweets100kfinalf.csv    # Processed dataset
├── model50000.pt           # Trained model (generated after running Model_training.py) #too big of a file to upload on github
└── requirements.txt        # Python package dependencies
```


