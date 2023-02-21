# %% Import
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from module import eda, cleaned_data, text_tokenization, text_pad_trunc, model_archi, model_train, plot_history, model_metrics, model_save

MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
OHE_PATH = os.path.join(os.getcwd(), 'saved_models', 'ohe.pkl')
TOKEN_PATH = os.path.join(os.getcwd(), 'saved_models', 'tokenizer.json')
MODEL_PNG_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.png')

SAVED_MODEL_PATH = os.path.join(os.getcwd(), 'saved_models')
if not os.path.exists(SAVED_MODEL_PATH):
    os.makedirs(SAVED_MODEL_PATH)
# %% Data Loading
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')
# %% Exploratory Data Analysis(EDA)
eda(df)
# %% Data Cleaning
text = df['text']
category = df['category']

text = cleaned_data(text)
# %% Data Preprocessing
# Data Vectorization/Tokenization
vocab_size = 5000
text_tokenized, tokenizer = text_tokenization(text, vocab_size)
# %%
# Padding & Truncating
maxlen = [len(i) for i in text_tokenized]
maxlen = np.array(maxlen)
maxlen = int(np.ceil(np.median(maxlen)))

text_tokenized = text_pad_trunc(text_tokenized, maxlen)
# %%
# OHE
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category, axis=-1))
# %%
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(text_tokenized, category, test_size=0.2, stratify=category, random_state=0)
# %% Model Development
num_labels = len(np.unique(y_train, axis=0))
embedding_dim = 64
drop_rate = 0.4

model = model_archi(vocab_size, num_labels, embedding_dim, drop_rate, MODEL_PNG_PATH)
# %%
hist, model = model_train(model, X_train, X_test, y_train, y_test, epochs=50)

# %% Model Analysis
plot_history(hist)

model_metrics(model, ohe, X_test, y_test, category)
# %% Model Deployment
model_save(MODEL_PATH, OHE_PATH, TOKEN_PATH, model, ohe, tokenizer)

# %%
