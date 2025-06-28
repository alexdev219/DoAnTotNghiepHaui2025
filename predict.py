import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load mô hình và tokenizer đã lưu
model = load_model("english_fake_news_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Sử dụng lại độ dài chuỗi tối đa như khi huấn luyện
MAX_LEN = 200

def predict_news(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    label = "REAL" if prediction >= 0.5 else "FAKE"
    print(f"\n📌 Prediction: {label} (Confidence Score: {prediction:.4f})\n")

# Vòng lặp kiểm tra liên tục
if __name__ == "__main__":
    print("🔍 Enter a news article or headline (type 'exit' to stop):")
    while True:
        text = input("📝 News Text: ")
        if text.strip().lower() == "exit":
            print("👋 Exiting prediction mode.")
            break
        if not text.strip():
            print("⚠️ Please enter non-empty text.")
            continue
        predict_news(text)
