from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd 
from sklearn.utils import resample
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def clean_tweets(tweets):
    tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub(r'\W', ' ', x))
    tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub(r'\^[a-zA-Z]\s+', ' ', x) )
    tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
    tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub(r'^b\s+', '', x))
    tweets['Tweet'] = tweets['Tweet'].apply(lambda x: x.lower())

np.random.seed(42)

tweets = pd.read_csv('train.csv', dtype = {'id':np.int64, 'Category':str, 'Tweet':str})

data_positive = tweets[tweets['Category'] == 'positive']
data_neutral = tweets[tweets['Category'] == 'neutral']
data_negative = tweets[tweets['Category'] == 'negative']


data_neutral_upsampled = resample(data_neutral, replace=True, n_samples=len(data_positive), random_state=0)
data_negative_upsampled = resample(data_negative, replace=True, n_samples=len(data_positive), random_state=0)

tweets = pd.concat([data_positive, data_neutral_upsampled, data_negative_upsampled])

clean_tweets(tweets)

texts = tweets.iloc[:, 2].values
labels = tweets.iloc[:, 1].values
for i,l in enumerate(labels):
    if l=='Tweet':
        texts = np.delete(texts,i,0)
        labels = np.delete(labels,i,0)
        break

vectorizer = TfidfVectorizer (max_features=2000, min_df=8, max_df=0.9, stop_words=stopwords.words('english'))
texts = vectorizer.fit_transform(texts).toarray()


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=0)

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

tweets_test = pd.read_csv('test.csv', dtype = {'Id':str, 'Tweet':str})
tweets_test = tweets_test.dropna()
tweets_test = tweets_test.astype({'Id': 'int64'})

clean_tweets(tweets_test)

texts_test = tweets_test.iloc[:, 1].values
ids = tweets_test.iloc[:, 0].values
X_test = vectorizer.transform(texts_test).toarray()

predictions = text_classifier.predict(X_test)

submission = pd.DataFrame({"Id":ids,"Category":predictions})
submission.to_csv('sumbission.csv', header=True, index=False, columns=["Id","Category"])