#I’m using NLTK (Natural Language Toolkit) here to work with human language data, and heapq to pull out the top sentences from a chunk of text.

from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords  # skips over common words like is, am, are
from nltk.tokenize import word_tokenize, sent_tokenize  # splits the text into sentences and words
from heapq import nlargest  # grabs the top 3 sentences

app = Flask(__name__)

# This is where the actual text summarizing happens
def summarize_text(text, num_sentences=3):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = {}
    for word in words:
        word = word.lower()
        if word not in stopWords:
            freqTable[word] = freqTable.get(word, 0) + 1

    maximum_frequency = max(freqTable.values()) if freqTable else 1
    for word in freqTable:
        freqTable[word] /= maximum_frequency

    sentence_scores = {}
    sentences = sent_tokenize(text)
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freqTable:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + freqTable[word]

    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# API endpoint for summarizing
@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']
    num_sentences = data.get('num_sentences', 3)

    summary = summarize_text(input_text, num_sentences)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    # Makes the app available on your network, not just your own computer
    app.run(host='0.0.0.0', port=5000)
