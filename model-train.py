import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# 1. Daten laden
with open("dataset-1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Zeichen und Mappings
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# 3. Sequenzen erstellen
maxlen = 40  # Länge der Sequenz
step = 3     # Schrittweite zum Erzeugen neuer Sequenzen
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i+maxlen])
print("Anzahl der Sequenzen:", len(sentences))

# 4. Daten in numerische Form bringen
X = np.zeros((len(sentences), maxlen), dtype=np.int32)
y = np.zeros((len(sentences)), dtype=np.int32)
for i, sentence in enumerate(sentences):
    X[i] = [char_to_idx[char] for char in sentence]
    y[i] = char_to_idx[next_chars[i]]

vocab_size = len(chars)

# 5. Modell erstellen
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

# 6. Modell trainieren
model.fit(X, y, batch_size=128, epochs=20)

