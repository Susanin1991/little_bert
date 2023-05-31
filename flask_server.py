import os
import sys
import socket

import transformers
from flask import Flask, request
import numpy
import torch
import pickle


app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    # Process the data as needed
    result = {'message': 'Data processed successfully'}
    return result

def process_text(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    max_len = len(tokenized)
    padded = numpy.array(tokenized + [0] * (max_len - len(tokenized)))
    input_ids = torch.tensor([padded])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return last_hidden_states[:, 0, :].numpy()


if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = (
        transformers.DistilBertModel,
        transformers.DistilBertTokenizer,
        'distilbert-base-uncased')

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Check if the trained model file exists
    model_file = 'resources//model.pkl'
    if os.path.isfile(model_file):
        print('Loading the trained model from', model_file)
        with open(model_file, 'rb') as f:
            lr_clf = pickle.load(f)
    else:
        print('model is missing')
        sys.exit()

    host = socket.gethostname()
    port = 5000
    app.run(host=host, port=port)