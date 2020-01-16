from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd 

np.random.seed(42)

tweets = pd.read_csv('train.csv', dtype = {'id':np.int64, 'Category':str, 'Tweet':str})
'''
for i,tweet in enumerate(tweets.values):
    if tweet[2]=='Not Available':
        tweets = tweets.drop(i)
'''     

texts = tweets.iloc[:, 2].values
labels = tweets.iloc[:, 1].values

vectorizer = TfidfVectorizer (max_features=2000, min_df=8, max_df=0.9, stop_words=stopwords.words('english'))
texts = vectorizer.fit_transform(texts).toarray()
'''
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(texts, labels)

tweets_test = pd.read_csv('test.csv', dtype = {'Id':str, 'Tweet':str})
tweets_test = tweets_test.dropna()
tweets_test = tweets_test.astype({'Id': 'int64'})

texts_test = tweets_test.iloc[:, 1].values
ids = tweets_test.iloc[:, 0].values
X_test = vectorizer.transform(texts_test).toarray()

predictions = text_classifier.predict(X_test)

submission = pd.DataFrame({"Id":ids,"Category":predictions})
submission.to_csv('sumbission.csv', header=True, index=False, columns=["Id","Category"])
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=0)

'''from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
'''
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 3), random_state=0)
clf.fit(X_train, y_train)
                    
predictions = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

from xgboost import XGBClassifier as XGBoostClassifier
classifier = XGBoostClassifier(seed=42,n_estimators=403,max_depth=10,objective="binary:logistic",learning_rate=0.15)

classifier.fit(X_train, y_train)
                    
predictions = classifier.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))