# bag of word model
# creating dataset
import pandas as pd
import json
import pickle
try:
    with open("../logs/panda_dataset.pkl", 'rb') as f_read:
        dataset = pickle.load(f_read)
except FileNotFoundError as file_error:
    with open("../sample-data/dataset_combined.json") as file_dataset_json:
        dataset_lines = file_dataset_json.readlines()
        dataset = pd.DataFrame(columns=["concor", 'annot'])
        for idx, line in enumerate(dataset_lines):
            json_item = json.loads(line)
            annotation = 1 if sum([int(annot) for annot in [json_item["annotator1"],
                                                            json_item["annotator2"], json_item["annotator3"]]
                                   ]) >= 2 else 0
            dataset.loc[idx] = [json_item["concordanceText"], annotation]
    with open("../logs/panda_dataset.pkl", 'wb') as f_write:
        pickle.dump(dataset, f_write)
        # print(dataset["annot"])
# creating the one-hot encoding for the words based
try:
    with open("../logs/vectorizer.pkl", 'rb') as f_read:
        vectorizer = pickle.load(f_read)
    with open("../logs/dataset_X.pkl", 'rb') as f_read:
        dataset_X = pickle.load(f_read)
    with open("../logs/dataset_y.pkl", 'rb') as f_read:
        dataset_y = pickle.load(f_read)

except FileNotFoundError as file_error:
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(lowercase=False)
    dataset_X = vectorizer.fit_transform(dataset["concor"].tolist())
    dataset_y = dataset['annot'].tolist()
    # print("Vocab: ")
    # print(vectorizer.vocabulary_)
    # print(dataset_X[0])
    # print(vectorizer.inverse_transform(dataset_X[0]))
    with open("../logs/vectorizer.pkl", 'wb') as f_write:
        pickle.dump(vectorizer, f_write)
    with open("../logs/dataset_X.pkl", 'wb') as f_write:
        pickle.dump(dataset_X, f_write)
    with open("../logs/dataset_y.pkl", 'wb') as f_write:
        pickle.dump(dataset_y, f_write)


# print(dataset_X[1].toarray())
# print(dataset_y[1])
# print(vectorizer.inverse_transform(dataset_X[1]))