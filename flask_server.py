import os
import sys
import socket
import transformers
import numpy
import torch
import pickle

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_data():
    message = request.get_json()
    message = bert_process(message)
    response = {'response': 'Bert result is: ' + message}
    return jsonify(response)


def process_text(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    max_len = len(tokenized)
    padded = numpy.array(tokenized + [0] * (max_len - len(tokenized)))
    input_ids = torch.tensor([padded])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return last_hidden_states[:, 0, :].numpy()


def bert_process(text):
    print('message is: ', text)
    input_features = process_text(text['message'])
    result = lr_clf.predict(input_features)
    data = ''
    if result == [0]:
        data = "BERT thinks it's bad thing to say"
        print(data, result)
    if result == [1]:
        data = "BERT thinks it's good thing to say"
        print(data, result)

    return data


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

    # host = "127.0.0.1"
    # port = 8000
    # app.run(host=host, port=port)
    app.run()