# Here I'm using nltk(Natural Language toolkit)to handle human language data and heapq to find top-ranked sentences----
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords #ignores the common words like is,am,are---
from nltk.tokenize import word_tokenize, sent_tokenize #break down the text into individual sentence s and words--
from heapq import nlargest #picks the top 3 sentences---

app = Flask(__name__)

# --- Text Summarization Logic is here---
def summarize_text(text, num_sentences=3):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = dict() # determines which words are most important in the text------
    for word in words:
        word = word.lower()
        if word not in stopWords:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

    maximum_frequency = max(freqTable.values()) if freqTable else 1
    for word in freqTable.keys():
        freqTable[word] = (freqTable[word]/maximum_frequency)

    sentence_scores = {}
    sentences = sent_tokenize(text)
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freqTable.keys():
                if sent not in sentence_scores:
                    sentence_scores[sent] = freqTable[word]
                else:
                    sentence_scores[sent] += freqTable[word]

    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# --- Flask API Endpoint ---
@app.route('/api/summarize', methods=['POST']) # accepts only POST requests--
def api_summarize():
    data = request.get_json() # It allows an optional num sentences parameter---
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']
    # Optional: get number of sentences from request, default to 3
    num_sentences = data.get('num_sentences', 3)

    summary = summarize_text(input_text, num_sentences)
    return jsonify({'summary': summary}) # after processing the JSON sends the summary back in clean JSON format

if __name__ == '__main__':
    # Run the app on host 0.0.0.0 for potential deployment/access from other devices
    app.run(host='0.0.0.0', port=5000)
