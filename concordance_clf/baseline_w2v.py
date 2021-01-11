import pickle
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# from torch.utils.data import DataLoader


try:
    # with open("../logs/vocab.pkl", 'rb') as f_read:
    #     vocabulary = pickle.load(f_read)
    with open("../logs/train_X.pkl", 'rb') as f_read:
        train_X = pickle.load(f_read)
    with open("../logs/train_y.pkl", 'rb') as f_read:
        train_y = pickle.load(f_read)
    with open("../logs/test_X.pkl", 'rb') as f_read:
        test_X = pickle.load(f_read)
    with open("../logs/test_y.pkl", 'rb') as f_read:
        test_y = pickle.load(f_read)

except FileNotFoundError as f_error:
    with open("../logs/panda_dataset.pkl", 'rb') as f_read:
        dataset = pickle.load(f_read)
        print(Counter(dataset["annot"].tolist()))

    dataset_X = dataset["concor"].tolist()
    dataset_y = dataset['annot'].tolist()
    train_X_raw, test_X_raw, train_y, test_y = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=1,
                                                                stratify=dataset_y)


    def text_pre_processor(text, tokenizer=RegexpTokenizer(r'\w+')):
        # text = str(text).replace("\n", "")
        # test = text.replace(',', '')
        text = tokenizer.tokenize(text)
        # remove stopwords
        output = []
        for token in text:
            if token not in stopwords.words("english"):
                output.append(token.lower())
        return output


    train_X = [text_pre_processor(text) for text in train_X_raw]
    test_X = [text_pre_processor(text) for text in test_X_raw]

    # # Create vocabulary/word index, to only include top 2000 words
    # en_tokenizer = get_tokenizer('spacy', language='en')
    #
    #
    # def build_vocab(text_list, tokenizer):
    #     counter = Counter()
    #     for text in text_list:
    #         counter.update(tokenizer(text))
    #     return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    #
    #
    # vocabulary = build_vocab(train_X, en_tokenizer)
    # print(len(vocabulary))
    #
    #
    # # replace words with their word_index in vocabulary
    # def create_index_tensor(text_list, vocabulary) -> object:
    #     data = []
    #     for text in text_list:
    #         tmp_tensor = torch.tensor([vocabulary[token] for token in en_tokenizer(text)], dtype=torch.long)
    #         data.append(tmp_tensor)
    #     return data
    #
    #
    # train_X = create_index_tensor(train_X, vocabulary)
    # test_X = create_index_tensor(test_X, vocabulary)
    # print(train_X_raw[0])
    # print(train_X[0])

    # with open("../logs/vocab.pkl", 'wb') as f_write:
    #     pickle.dump(vocabulary, f_write)
    with open("../logs/train_X.pkl", 'wb') as f_write:
        pickle.dump(train_X, f_write)
    with open("../logs/train_y.pkl", 'wb') as f_write:
        pickle.dump(train_y, f_write)
    with open("../logs/test_X.pkl", 'wb') as f_write:
        pickle.dump(test_X, f_write)
    with open("../logs/test_y.pkl", 'wb') as f_write:
        pickle.dump(test_y, f_write)

# # batching and padding the text
# BATCH_SIZE = 100  # number for conconrdence samples in a batch
# PAD_IDX = vocabulary['<pad>']
# BOS_IDX = vocabulary['<bos>']
# EOS_IDX = vocabulary['<eos>']
#
#
# # def generate_batch(cc_batch):
# #     batch = []
# #     for concordence in cc_batch:
# #         batch.append(torch.cat([torch.tensor([BOS_IDX]), concordence, torch.tensor([EOS_IDX])], dim=0))
# #     batch = pad_sequence(batch, padding_value=PAD_IDX)
# #     return batch
# #
# #
# # train_iter = DataLoader(train_X, batch_size=BATCH_SIZE,
# #                         shuffle=True, collate_fn=generate_batch)
# #
# # test_iter = DataLoader(test_X, batch_size=BATCH_SIZE,
# #                        shuffle=True, collate_fn=generate_batch)

# print(train_X[1])
# print(train_y[1])

# Using Stanford GloVe for word embedding:
CC_MEX_LEN = 11
w2v_dim = 300
w2v_dict = {}
try:
    with open('../w2v/w2v_dict.pkl', 'rb') as f_read:
        w2v_dict = pickle.load(f_read)
except FileNotFoundError as f_error:
    with open("../w2v/glove.6B.300d.txt") as f_read:
        for line in f_read:
            values = line.split()
            word = values[0]
            w2v_dict[word] = np.asarray(values[1:], dtype='float32')

    with open('../w2v/w2v_dict.pkl', 'wb') as f_write:
        pickle.dump(w2v_dict, f_write)


# print(w2v_dict['math'])

def cc_w2v(concordence, max_len=CC_MEX_LEN, w2v_dim=300):
    cc_vec = []
    pad_size = max_len - len(concordence)
    for token in concordence:
        try:
            word_vec = w2v_dict[token]
            cc_vec.append(word_vec)
        except KeyError as k_error:
            unk = np.zeros(w2v_dim)
            cc_vec.append(unk)

    if pad_size == 0:
        return np.array(cc_vec).reshape(1, -1)
    elif pad_size > 0:
        padding = np.zeros((pad_size, w2v_dim))
        cc_vec = np.append(np.array(cc_vec), padding, axis=0)
        return cc_vec.reshape(1, -1)


# for cc_token in train_X:
#     cc_w2v(cc_idxs)
train_X_emb = cc_w2v(train_X[0])
train_y_emb = np.array(train_y[0])

for idx, cc_tokens in enumerate(train_X[1:]):
    try:
        train_X_emb = np.append(train_X_emb, cc_w2v(cc_tokens), axis=0)
        train_y_emb = np.append(train_y_emb, train_y[idx])
    except Exception as error:
        pass


test_X_emb = cc_w2v(test_X[0])
test_y_emb = np.array(test_y[0])

for idx, cc_tokens in enumerate(test_X[1:]):
    try:
        test_X_emb = np.append(test_X_emb, cc_w2v(cc_tokens), axis=0)
        test_y_emb = np.append(test_y_emb, test_y[idx])
    except Exception as error:
        pass


print(train_X_emb.shape)
print(test_X_emb.shape)

from sklearn.linear_model import SGDClassifier
from sklearn import metrics

clf_svm = SGDClassifier(loss='hinge', alpha=1e-3, random_state=1)
clf_svm.fit(train_X_emb, train_y_emb)
prediction = clf_svm.predict(test_X_emb)
accuracy = metrics.accuracy_score(test_y_emb, prediction)
f1_score = metrics.f1_score(test_y_emb, prediction, average='macro')
conf_mat = metrics.confusion_matrix(test_y_emb, prediction, labels=[1, 0])

print("SVM:---------------")
print(f"Total Accuracy: {accuracy}")
print(f"f1_score: {f1_score}")
print(f'confusion matrix labels[1,0]:\n{conf_mat}')
