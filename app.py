import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the text data to recreate tokenizer
with open('hamlet.txt', 'r') as file:
    text = file.read().lower()

# Recreate tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
max_sequence_len = 14

# Rebuild model architecture EXACTLY as it was trained
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile to build the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Build model by calling predict with sample data
sample_input = np.zeros((1, max_sequence_len-1), dtype=np.int32)
try:
    _ = model.predict(sample_input, verbose=0)
except:
    pass

# Load the weights
model.load_weights('model_weights.h5')

# Function to predict top K words with probabilities
def predict_top_words(model, tokenizer, text, max_sequence_len, top_k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]
    
    # Get top K predictions
    top_indices = np.argsort(predicted)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                results.append((word, predicted[idx]))
                break
    return results

# Function to predict with temperature for more variety
def predict_next_word_temperature(model, tokenizer, text, max_sequence_len, temperature=0.7):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]
    
    # Apply temperature
    predicted = np.log(predicted + 1e-10) / temperature
    predicted = np.exp(predicted) / np.sum(np.exp(predicted))
    
    # Sample from distribution instead of argmax
    predicted_word_index = np.random.choice(len(predicted), p=predicted)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("🎭 Next Word Prediction With LSTM")
st.markdown("*Trained on Shakespeare's Hamlet*")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
prediction_mode = st.sidebar.radio(
    "Prediction Mode:",
    ["Top 5 Predictions", "Single Prediction (Creative)"]
)

if prediction_mode == "Single Prediction (Creative)":
    temperature = st.sidebar.slider(
        "Temperature (Creativity)", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.7, 
        step=0.1,
        help="Lower = more predictable, Higher = more creative"
    )

# Main input
input_text = st.text_input(
    "Enter the sequence of words:", 
    "To be or not to",
    help="Type any English text. Works best with Shakespeare-style phrases!"
)

# Predict button
if st.button("🔮 Predict Next Word", type="primary"):
    if input_text.strip():
        with st.spinner("Predicting..."):
            if prediction_mode == "Top 5 Predictions":
                predictions = predict_top_words(model, tokenizer, input_text, max_sequence_len, top_k=5)
                st.success("**Top 5 Predictions:**")
                for i, (word, prob) in enumerate(predictions, 1):
                    st.write(f"{i}. **{word}** — *{prob*100:.2f}% confidence*")
            else:
                next_word = predict_next_word_temperature(model, tokenizer, input_text, max_sequence_len, temperature)
                st.success(f"**Predicted next word:** {next_word}")
                st.info(f"Full text: *{input_text} {next_word}*")
    else:
        st.warning("⚠️ Please enter some text first!")

# Examples section
with st.expander("💡 Try these examples"):
    st.markdown("""
    - `to be or not`
    - `good night sweet`
    - `the king is`
    - `what is your`
    - `something is rotten in`
    - `my lord`
    """)

# Footer
st.markdown("---")
st.markdown("*Built with LSTM Neural Network | Trained on Shakespeare's Hamlet*")