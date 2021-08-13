import pandas as pd
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
import joblib

df_train = pd.read_csv('train.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
df_test = pd.read_csv('test.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
X_train = df_train['text']
Y_train = df_train['label']
X_test = df_test['text']
Y_test = df_test['label']

# from sklearn.naive_bayes import MultinomialNB
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, Y_train)
# print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))

# vectorizer = CountVectorizer()
# transformer = TfidfTransformer()
# tfidf_before = vectorizer.fit_transform(X_train)
# tfidf = transformer.fit_transform(tfidf_before)
# X_train_counts = vectorizer.fit_transform(X_train)
# X_train_tfidf = transformer.fit_transform(X_train_counts)

# svm_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
#                         ('clf', MultinomialNB())])
# clf = svm_pipeline.fit(X_train_tfidf, Y_train)

#print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# model.fit(X_train, Y_train)
# print('some score', model.score(X_test, Y_test))
# x = 'some text'
# print('predicted', model.predict([x]))
# pipeline_file = open("naive_model.sav","wb")
# joblib.dump(model,pipeline_file)
# pipeline_file.close()
#model = joblib.load(open('models/naive_model.sav', 'rb'))
#x = 'some text'
#print('some score', model.score(X_test, Y_test))
#print('predicted', model.predict([x]))

def predict_label(text, pipeline):
    result = pipeline.predict([text])
    return result

def get_probability(text, pipeline):
    result = pipeline.predict_proba([text])
    return result

# pipeline = joblib.load(open('models/svc_classifier.sav', 'rb'))
# text = "some_text"
# prediction = predict_label(text, pipeline)
# probability = get_probability(text, pipeline)
# print('prediction', prediction)
# print('probabilities', probability)

df = pd.read_csv('reports/svc_report.csv', index_col = 0)
print(df)