# bag of word model
# creating dataset
import pandas as pd
import json
import pickle
from collections import Counter
# from collections import Counter
try:
    with open("../logs/panda_dataset.pkl", 'rb') as f_read:
        dataset = pickle.load(f_read)
        print(Counter(dataset["annot"].tolist()))
except FileNotFoundError as file_error:
    with open("../data/combined_data.json") as file_dataset_json:
        dataset_lines = file_dataset_json.readlines()
        dataset = pd.DataFrame(columns=["concor", 'annot'])
        for idx, line in enumerate(dataset_lines):
            json_item = json.loads(line)
            ## Error in annotation
            annotation_list = [json_item["annotator1"], json_item["annotator2"], json_item["annotator3"]]
            # two annotation -1: do not include
            # -1, 0, 1 -> do not include
            # -1, 0, 0 -> include
            # -1, 1, 1 -> include
            annotation_counter = Counter(annotation_list)
            if annotation_counter[-1] == 2:
                continue
            elif annotation_counter[-1] == 1:
                if annotation_counter[0] == annotation_counter[1]:
                    continue
                else:
                    if annotation_counter[1]:
                        annotation = 1
                    elif annotation_counter[0]:
                        annotation = 0
            else:
                annotation = 1 if sum([int(annot) for annot in annotation_list]) >= 2 else 0

            dataset.loc[idx] = [json_item["concordanceText"], annotation]
    print(Counter(dataset["annot"].tolist()))
    with open("../logs/panda_dataset.pkl", 'wb') as f_write:
        pickle.dump(dataset, f_write)
        # print(dataset["annot"])
# creating the one-hot encoding for the words based
try:
    with open("../logs/train_X.pkl", 'rb') as f_read:
        train_X = pickle.load(f_read)
    with open("../logs/train_y.pkl", 'rb') as f_read:
        train_y = pickle.load(f_read)
    with open("../logs/test_X.pkl", 'rb') as f_read:
        test_X = pickle.load(f_read)
    with open("../logs/test_y.pkl", 'rb') as f_read:
        test_y = pickle.load(f_read)

except FileNotFoundError as file_error:
    # from sklearn.feature_extraction.text import CountVectorizer
    # vectorizer = CountVectorizer(lowercase=False)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words={'english'})
    from sklearn.model_selection import train_test_split
    dataset_X = dataset["concor"].tolist()
    dataset_y = dataset['annot'].tolist()
    train_X, test_X, train_y, test_y = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=1,
                                                        stratify=dataset_y)
    train_X = vectorizer.fit_transform(train_X)
    test_X = vectorizer.transform(test_X)
    # print("Vocab: ")
    # print(vectorizer.vocabulary_)
    # print(dataset_X[0])
    # print(vectorizer.inverse_transform(dataset_X[0]))
    with open("../logs/train_X.pkl", 'wb') as f_write:
        pickle.dump(train_X, f_write)
    with open("../logs/train_y.pkl", 'wb') as f_write:
        pickle.dump(train_y, f_write)
    with open("../logs/test_X.pkl", 'wb') as f_write:
        pickle.dump(test_X, f_write)
    with open("../logs/test_y.pkl", 'wb') as f_write:
        pickle.dump(test_y, f_write)

# print(dataset_X[1].toarray())
# print(dataset_y[1])
# print(vectorizer.inverse_transform(dataset_X[1]))


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB(alpha=0.1)
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)
accuracy = metrics.accuracy_score(test_y, prediction)
f1_score = metrics.f1_score(test_y, prediction, average='macro')
conf_mat = metrics.confusion_matrix(test_y, prediction, labels=[1, 0] )

print("Naive Bayes:---------------")
print(f"Total Accuracy: {accuracy}")
print(f"f1_score: {f1_score}")
print(f'confusion matrix labels[1,0]:\n{conf_mat}')

from sklearn.linear_model import SGDClassifier
clf_svm = SGDClassifier(loss='hinge', alpha=1e-3, random_state=1)
clf_svm.fit(train_X, train_y)
prediction = clf_svm.predict(test_X)
accuracy = metrics.accuracy_score(test_y, prediction)
f1_score = metrics.f1_score(test_y, prediction, average='macro')
conf_mat = metrics.confusion_matrix(test_y, prediction, labels=[1, 0] )

print("SVM:---------------")
print(f"Total Accuracy: {accuracy}")
print(f"f1_score: {f1_score}")
print(f'confusion matrix labels[1,0]:\n{conf_mat}')