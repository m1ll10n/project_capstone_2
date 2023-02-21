import numpy as np
import matplotlib.pyplot as plt
import re, os, datetime, pickle, json

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

def eda(df):
    """Perform EDA on dataset. Check for paragraphs

    Args:
        df (DataFrame): Dataset
    """
    print(df.head())
    print(df['text'][1])

def cleaned_data(text):
    """Cleaning text for training/testing model.
    First cleaning is to remove everything in between () brackets (i.e. (£5.8bn) to '')
    Second cleaning is to replace everything except for letters into ' '
    Last cleaning is to remove any singular characters (i.e. 'firm s shares' to 'firm shares')

    Args:
        text (Series): Article's text

    Returns:
        Series: Cleaned article's text
    """
    for i, data in enumerate(text):
        temp = re.sub('\(.*?\)', ' ', data) # remove everything in () brackets
        temp = re.sub('[^a-zA-Z]', ' ', temp)
        temp = re.sub('\s[a-z]\\b', '', temp) # firm s shares = firms shares
        text[i] = temp.lower()
    return text

def text_tokenization(text, vocab_size):
    """Text Tokenization/Vectorization

    Args:
        text (Series): Article's text
        vocab_size (int): Vocabulary size

    Returns:
        ndarray: Tokenized text
        Tokenizer: Tokenizer
    """
    oov_token = '<OOV>'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(text)

    word_index = tokenizer.word_index
    print(list(word_index.items())[0:20])

    text_tokenized = tokenizer.texts_to_sequences(text)

    return text_tokenized, tokenizer

def text_pad_trunc(text_tokenized, maxlen):
    """Text Padding & Truncation

    Args:
        text_tokenized (list): Tokenized text
        maxlen (int): Maximum length for all sequences

    Returns:
        ndarray: Padded & truncated text
    """
    text_tokenized = pad_sequences(text_tokenized, maxlen=maxlen, padding='post', truncating='post')
    text_tokenized = np.expand_dims(text_tokenized, axis=-1)

    return text_tokenized

def model_archi(vocab_size, num_labels, embedding_dim, drop_rate, MODEL_PNG_PATH):
    """Model architecture

    Args:
        vocab_size (int): Vocabulary size
        num_labels (int): Number of output classes
        embedding_dim (int): Embedding dimensions
        drop_rate (float): Dropout rate
        MODEL_PNG_PATH (str): Path to save model.png

    Returns:
        Sequential: Sequential model
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(embedding_dim, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(LSTM(embedding_dim))
    model.add(Dropout(drop_rate))
    model.add(Dense(num_labels, activation='softmax'))

    model.summary()
    plot_model(model, to_file=MODEL_PNG_PATH, show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_train(model, X_train, X_test, y_train, y_test, epochs=30):
    """Training the model

    Args:
        model (Sequential): Sequential model
        X_train (ndarray): Feature training variable
        X_test (ndarray): Feature testing variable
        y_train (ndarray): Target training variable
        y_test (ndarray): Target testing variable
        epochs (int, optional): Number of training epochs. Defaults to 30.

    Returns:
        History: Training history
        Sequential: Sequential model
    """
    log_dir = os.path.join(os.getcwd(), 'tensorboard_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = TensorBoard(log_dir=log_dir)

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, callbacks=[tb_callback])

    return hist, model

def plot_history(hist):
    """Plotting training loss and training accuracy

    Args:
        hist (History): Training history
    """
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['training loss', 'validation loss'])
    plt.show()

    plt.figure()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.show()

def model_metrics(model, ohe, X_test, y_test, category):
    """To display accuracy, f1_score, classification report, and confusion matrix

    Args:
        model (Sequential): Sequential model
        ohe (OneHotEncoder): OneHotEncoder model
        X_test (ndarray): Feature testing variable
        y_test (ndarray): Target testing variable
        category (ndarray): Target variable
    """
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    acc_scr = accuracy_score(y_test, y_pred)
    f1_scr = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy Score: {acc_scr}, F1 Score: {f1_scr}\n\n")
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(cr)

    labels = ohe.inverse_transform(np.unique(category, axis=0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def model_save(MODEL_PATH, OHE_PATH, TOKEN_PATH, model, ohe, tokenizer):
    """To save model.h5, ohe.pkl, tokenizer.json

    Args:
        MODEL_PATH (str): Path to save model.h5
        OHE_PATH (str): Path to save ohe.pkl
        TOKEN_PATH (str): Path to save tokenizer.json
        model (Sequential): Sequential model
        ohe (OneHotEncoder): OneHotEncoder model
        tokenizer (Tokenizer): Tokenizer model
    """
    model.save(MODEL_PATH)

    with open(OHE_PATH, 'wb') as f:
        pickle.dump(ohe, f)

    token_json = tokenizer.to_json()
    with open(TOKEN_PATH, 'w') as f_json:
        json.dump(token_json, f_json)