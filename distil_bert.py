import sys

import numpy
import torch
import transformers
import socket
import os
import pickle


model_class, tokenizer_class, pretrained_weights = (
    transformers.DistilBertModel,
    transformers.DistilBertTokenizer,
    'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


def process_text(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    max_len = len(tokenized)
    padded = numpy.array(tokenized + [0] * (max_len - len(tokenized)))
    input_ids = torch.tensor([padded])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return last_hidden_states[:, 0, :].numpy()


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

server_socket = socket.socket()
server_socket.bind((host, port))
server_socket.listen(2)

print('Listening for incoming connections...')
conn, address = server_socket.accept()

while True:

    text = conn.recv(1024).decode()
    print('Connected to', address)
    print('Received text:', text)

    if text.lower().strip() == 'exit':
        conn.close()
        break

    input_features = process_text(text)
    result = lr_clf.predict(input_features)
    data = ''
    if result == [0]:
        data = "BERT thinks it's bad thing to say"
        print(data, result)
    if result == [1]:
        data = "BERT thinks it's good thing to say"
        print(data, result)

    conn.send(data.encode())  # send data to the client

