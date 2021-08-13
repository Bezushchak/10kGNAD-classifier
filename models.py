import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.pipeline import make_pipeline, Pipeline
import joblib

def get_the_data():
    df_train = pd.read_csv('data/train.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
    df_test = pd.read_csv('data/test.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
    X_train = df_train['text']
    Y_train = df_train['label']
    X_test = df_test['text']
    Y_test = df_test['label']
    return X_train, Y_train, X_test, Y_test

def naive_bayes_pipeline(X_train, Y_train):
    naive_bayes_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    naive_bayes_pipeline.fit(X_train, Y_train)
    return naive_bayes_pipeline

def svm_pipeline(X_train, Y_train):
    # count_vect=CountVectorizer()
    # X_train_counts =count_vect.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf =tfidf_transformer.fit_transform(X_train_counts)
    svm = LinearSVC()
    svm_pipeline = make_pipeline(TfidfVectorizer(), CalibratedClassifierCV(svm))
    svm_pipeline.fit(X_train, Y_train)
    return svm_pipeline

def log_reg_pipeline(X_train, Y_train):
    log_reg_pipeline = make_pipeline(CountVectorizer(),LogisticRegression())
    log_reg_pipeline.fit(X_train,Y_train)
    return log_reg_pipeline

def store_pipeline(pipeline, name):
    pipeline_file = open(name + ".sav","wb")
    joblib.dump(pipeline,pipeline_file)
    pipeline_file.close()

def play_the_model(pipeline, X_test, Y_test):
    predictions = pipeline.predict(X_test)
    return pipeline, predictions

def gen_classification_report(name, Y_test, predictions):
    report = metrics.classification_report(Y_test, predictions[1], output_dict = True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(str(name)+"_report.csv", index = True)

def main():

    #getting the data
    X_train, Y_train, X_test, Y_test = get_the_data()

    #Naive Bayes classifier
    nb_pipeline = naive_bayes_pipeline(X_train, Y_train)
    predictions = play_the_model(nb_pipeline, X_test, Y_test)
    print('naive bayes worked')
    store_pipeline(nb_pipeline, "naive_bayes_classifier")
    print ('naive bayes stored')

    print ("Naive Bayes Classification Report")
    print(metrics.classification_report(Y_test,predictions[1]))
    gen_classification_report('naive_bayes', Y_test, predictions)
    print(metrics.accuracy_score(Y_test,predictions[1]))

    #Support Vector classifier
    svc_pipeline = svm_pipeline(X_train, Y_train)
    predictions = play_the_model(svc_pipeline, X_test, Y_test)
    print ('svc worked')
    store_pipeline(svc_pipeline, "svc_classifier")
    print('svc stored')

    print ("SVM Classification Report")
    print(metrics.classification_report(Y_test,predictions[1]))
    gen_classification_report('svc', Y_test, predictions)
    print(metrics.accuracy_score(Y_test,predictions[1]))

    # Log Regression classifier
    lr_pipeline = log_reg_pipeline(X_train, Y_train)
    predictions = play_the_model(svc_pipeline, X_test, Y_test)
    print('log reg worked')
    store_pipeline(lr_pipeline, "log_reg_classifier")
    print('log reg stored')

    print ("Log Reg Classification Report")
    print(metrics.classification_report(Y_test,predictions[1]))
    gen_classification_report('log_reg', Y_test, predictions)
    print(metrics.accuracy_score(Y_test,predictions[1]))
    
if __name__ == '__main__':
    main()