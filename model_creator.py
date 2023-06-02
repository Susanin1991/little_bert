import numpy
import pandas
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pandas.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)

# load 5 first lines
df.head()

model_class, tokenizer_class, pretrained_weights = (
    transformers.DistilBertModel,
    transformers.DistilBertTokenizer,
    'distilbert-base-uncased')

## BERT instead of distilBERT:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Загрузка предобученной модели/токенизатора
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Function to process input text
def process_text(text):
    # Tokenize the text
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    max_len = len(tokenized)

    # Pad the tokenized input
    padded = numpy.array(tokenized + [0] * (max_len - len(tokenized)))

    # Convert to torch tensor
    input_ids = torch.tensor([padded])

    # Get the last hidden states from BERT
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    # Return the features
    return last_hidden_states[:, 0, :].numpy()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

#tokenize
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = numpy.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

input_ids = torch.tensor(numpy.array(padded))
# input_ids = input_ids.to(device)

with torch.no_grad():
    last_hidden_states = model(input_ids)

features = last_hidden_states[0][:,0,:].numpy()
labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
res = lr_clf.score(test_features, test_labels)
print('score is = ', str(res))
