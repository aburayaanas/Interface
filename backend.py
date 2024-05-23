import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat import textstat
from collections import Counter
from nltk import pos_tag, word_tokenize, ngrams
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

# Load BERT tokenizer and model
print("Loading BERT tokenizer and model...")
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load your model here
print("Loading custom model...")
model = tf.keras.models.load_model('colab_model.h5')

# Function to calculate n-gram repetitiveness
def ngram_repetitiveness(text, n=2):
    words = text.split()
    n_grams = list(ngrams(words, n))
    ngram_counts = Counter(n_grams)
    repeats = {ngram: count for ngram, count in ngram_counts.items() if count > 1}
    repetitiveness_score = sum(repeats.values()) / len(n_grams) if n_grams else 0
    return repetitiveness_score

# Function to calculate POS repetitiveness
def pos_repetitiveness(pos_tags):
    tag_sequences = [tag for _, tag in pos_tags]
    tag_counts = Counter(tag_sequences)
    repeats = {tag: count for tag, count in tag_counts.items() if count > 1}
    repetitiveness_score = sum(repeats.values()) / len(tag_sequences) if tag_sequences else 0
    return repetitiveness_score

def preprocess_text(text):
    # Get sentiment score
    sentiment_score = analyzer.polarity_scores(text)['compound']

    # Tokenize input text
    input_ids = tokenizer(text, padding=True, truncation=True, return_tensors="tf")["input_ids"]

    # Get BERT embeddings
    outputs = bert_model(input_ids)
    pooled_output = outputs.pooler_output.numpy()

    # Calculate readability metrics
    readability_metrics = textstat.flesch_reading_ease(text), \
                           textstat.flesch_kincaid_grade(text), \
                           textstat.automated_readability_index(text), \
                           textstat.coleman_liau_index(text), \
                           textstat.smog_index(text), \
                           textstat.gunning_fog(text)

    # Calculate n-gram repetitiveness
    bigram_repetitiveness = ngram_repetitiveness(text, n=2)

    # Perform POS tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    pos_tag_freq = Counter(tag for _, tag in pos_tags)

    # Define all possible POS tags
    all_pos_tags = ['NNP', 'CC', '(', 'CD', 'VBD', ')', 'DT', 'JJ', 'NN', '.', 'PRP$', 'IN', 'POS', ',', 'WRB', 'PRP', 'NNS', 'WDT', 'VBZ', 'VBN', 'VBG', 'RB', 'VB', 'SYM', ':', 'MD', '$', 'JJS', 'VBP', 'FW', 'WP', 'TO', 'RP', '``', 'NNPS', 'UH', 'JJR', 'RBS', 'EX', 'PDT', 'RBR', 'WP$', '#', 'LS']
    # Ensure all required POS tag columns are present
    pos_tags_input = np.array([pos_tag_freq.get(tag, 0) for tag in all_pos_tags]).reshape(1, -1)

    # Calculate POS structural repetitiveness
    pos_rep_score = pos_repetitiveness(pos_tags)

    # Prepare input data for prediction
    input_data = {
        'bert_input': pooled_output,
        'sentiment_input': np.array([[sentiment_score]]),
        'repetitiveness_input': np.array([[bigram_repetitiveness, pos_rep_score]]),
        'pos_tags_input': pos_tags_input
    }

    return input_data

def predict(text):
    try:
        # Preprocess the input text
        input_data = preprocess_text(text)
        print("Input data for prediction:", input_data)  # Debugging line
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        print("Error during prediction function:", e)
        print(traceback.format_exc())
        raise e

# Function to perform OCR on the image
def perform_ocr(image_path):
    try:
        extracted_text = pytesseract.image_to_string(Image.open(image_path))
        return extracted_text
    except Exception as e:
        print("Error during OCR:", e)
        return None

def preprocess_input(input_data):
    if isinstance(input_data, str):
        # Input is text, preprocess it directly
        return preprocess_text(input_data)
    else:
        # Input is an image file, perform OCR
        extracted_text = perform_ocr(input_data)
        if extracted_text:
            # If OCR extracted text, preprocess it
            return preprocess_text(extracted_text)
        else:
            # If OCR failed, return None
            return None

@app.route('/predict', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging line
        text = data.get('text')
        if text:
            predictions = predict(text)
            return jsonify(predictions.tolist())
        return jsonify({'error': 'No text provided'}), 400
    except Exception as e:
        print("Error during prediction:", e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        image_path = data.get('imagePath')
        if image_path:
            extracted_text = perform_ocr(image_path)
            if extracted_text:
                predictions = predict(extracted_text)
                return jsonify(predictions.tolist())
            return jsonify({'error': 'OCR failed to extract text'}), 400
        return jsonify({'error': 'No image path provided'}), 400
    except Exception as e:
        print("Error during prediction:", e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
