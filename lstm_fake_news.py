import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# Cài đặt
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100

# Load dữ liệu
def load_data(fake_path, real_path):
    fake_df = pd.read_csv(fake_path, low_memory=False)
    real_df = pd.read_csv(real_path)

    fake_df['label'] = 0
    real_df['label'] = 1

    data = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)
    texts = data['title'] + " " + data['text']
    labels = data['label'].values

    return texts, labels

# Tiền xử lý dữ liệu
def preprocess(texts, labels):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    return padded, np.array(labels)

# Xây dựng mô hình
def build_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Chạy toàn bộ pipeline
def run_pipeline(fake_path, real_path):
    print(" Loading data...")
    texts, labels = load_data(fake_path, real_path)

    X, y = preprocess(texts, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    print(" Building model...")
    model = build_model()

    print(" Training model...")
    model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=32)

    print(" Saving model...")
    model.save("english_fake_news_model.h5")

    print(" Evaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Gọi pipeline
run_pipeline("Fake.csv", "True.csv")
