import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
df_train = pd.read_csv('Corona(1)_NLP_train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('Corona_NLP_test.csv')
df_train = df_train[['OriginalTweet', 'Sentiment']]
df_test = df_test[['OriginalTweet', 'Sentiment']]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df_train['OriginalTweet'])
y_train = df_train['Sentiment']
X_test = vectorizer.transform(df_test['OriginalTweet'])
y_test = df_test['Sentiment']
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)
y_pred = naive_bayes_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy*100,3))
print("Classification Report:")
print(classification_report(y_test, y_pred))
input_text = "I'm feeling really optimistic about the future despite the challenges we're facing due to the pandemic. #StayPositive"

input_text_processed = vectorizer.transform([input_text])

predicted_sentiment = naive_bayes_model.predict(input_text_processed)[0]

print("Predicted Sentiment:", predicted_sentiment)
