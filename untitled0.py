I’m using NLTK (Natural Language Toolkit) to mess around with text—breaking it up, skipping over boring words, and all that. I also bring in heapq because it’s great for grabbing the best sentences out of a big chunk of writing.

from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords  # skips over words like “is” and “the”
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest  # picks out the top sentences

app = Flask(__name__)

# Here’s where the main action happens: turning a big block of text into a short summary.
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

# API endpoint that does the summarizing
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
    # This makes the app show up on your network, not just your PC
    app.run(host='0.0.0.0', port=5000)
