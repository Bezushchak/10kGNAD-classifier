import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib

#get the data
def get_the_data():
    df_train = pd.read_csv('train.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
    df_test = pd.read_csv('test.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
    X_train = df_train['text']
    Y_train = df_train['label']
    X_test = df_test['text']
    Y_test = df_test['label']
    return X_train, Y_train, X_test, Y_test

#fit the model


#save the model
def store_pipeline(pipeline, name):
    pipeline_file = open(name + ".pkl","wb")
    joblib.dump(pipeline,pipeline_file)
    pipeline_file.close()

def main():
    X_train, Y_train, X_test, Y_test = get_the_data()
    lr_pipeline = log_reg_pipeline(X_train, Y_train)
    print('The Log Reg score is ', lr_pipeline.score(X_test,Y_test))
    store_pipeline(lr_pipeline)

if __name__ == '__main__':
    main()