import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# ----------------------------
# Prepare training data
# ----------------------------
text = "hi hello, how are you? hello hi"

# Tokenize each character
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1  # include 0 padding

# Create sequences (previous 3 characters -> next character)
seq_length = 3
sequences = []
for i in range(seq_length, len(text)):
    seq = text[i-seq_length:i+1]
    sequences.append(seq)

# Separate input and output
X = [tokenizer.texts_to_sequences([seq[:-1]])[0] for seq in sequences]
y = [tokenizer.texts_to_sequences([seq[-1]])[0][0] for seq in sequences]

# Pad sequences (not strictly needed here since all same length)
max_seq_len = seq_length
X = pad_sequences(X, maxlen=max_seq_len, padding='pre')
y = to_categorical(y, num_classes=total_chars)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ----------------------------
# Build the model
# ----------------------------
model = Sequential()
model.add(Embedding(input_dim=total_chars, output_dim=32, input_length=max_seq_len))
model.add(LSTM(32))
model.add(Dense(total_chars, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------
# Train the model
# ----------------------------
model.fit(X, y, epochs=200, verbose=2)

# ----------------------------
# Function to generate text
# ----------------------------
def generate_text(model, tokenizer, seed_text, length=10):
    result = seed_text
    for _ in range(length):
        encoded = tokenizer.texts_to_sequences([seed_text])  # encode seed
        encoded = pad_sequences(encoded, maxlen=max_seq_len, padding='pre')
        pred_probs = model.predict(encoded, verbose=0)
        predicted_index = np.argmax(pred_probs)
        next_char = tokenizer.index_word[predicted_index]
        result += next_char
        seed_text = seed_text[1:] + next_char  # slide window
    return result

# ----------------------------
# Test text generation
# ----------------------------
seed = "h"
generated = generate_text(model, tokenizer, seed_text=seed, length=20)
print("Generated text:", generated)
