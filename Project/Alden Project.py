#1)Cancer Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

# load dataset
#NOTE: here i have renamed the dataset as breast-cancer.data.csv
df=pd.read_csv('breast-cancer.data.csv')
print(df.head())
# split features and target
X=df.iloc[:, 1:].values
y=df.iloc[:, 0].values

encoder=LabelEncoder()
X[:,0]=encoder.fit_transform(X[:,0])
X[:,1]=encoder.fit_transform(X[:,1])
X[:,2]=encoder.fit_transform(X[:,2])
X[:,3]=encoder.fit_transform(X[:,3])
X[:,4]=encoder.fit_transform(X[:,4])
X[:,6]=encoder.fit_transform(X[:,6])
X[:,7]=encoder.fit_transform(X[:,7])
X[:,8]=encoder.fit_transform(X[:,8])

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#feature selection
estimator=LogisticRegression()
selector=RFE(estimator, n_features_to_select=5, step=1)
selector=selector.fit(X_train, y_train)
X_train=selector.transform(X_train)
X_test=selector.transform(X_test)

#Logestic Regression
param_grid_lr={'C': [0.01, 0.1, 1, 10, 100]}
lr=LogisticRegression(random_state=42)
grid_search_lr=GridSearchCV(lr, param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)
lr_best=grid_search_lr.best_estimator_

#KNN
param_grid_knn={'n_neighbors': [3, 5, 7, 9]}
knn=KNeighborsClassifier()
grid_search_knn=GridSearchCV(knn, param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)
knn_best=grid_search_knn.best_estimator_

#Naive Bayes model
nb=GaussianNB()
nb.fit(X_train, y_train)

# predict on test set
y_pred_lr=lr_best.predict(X_test)
y_pred_knn=knn_best.predict(X_test)
y_pred_nb=nb.predict(X_test)

# evaluate performance
acc_lr=accuracy_score(y_test, y_pred_lr)
acc_knn=accuracy_score(y_test, y_pred_knn)
acc_nb=accuracy_score(y_test, y_pred_nb)

cm_lr=confusion_matrix(y_test, y_pred_lr)
cm_knn=confusion_matrix(y_test, y_pred_knn)
cm_nb=confusion_matrix(y_test, y_pred_nb)

cr_lr=classification_report(y_test, y_pred_lr)
cr_knn=classification_report(y_test, y_pred_knn)
cr_nb=classification_report(y_test, y_pred_nb)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.heatmap(cm_lr, annot=True, cmap='Blues', ax=axes[0])
sns.heatmap(cm_knn, annot=True, cmap='Blues', ax=axes[1])
sns.heatmap(cm_nb, annot=True, cmap='Blues', ax=axes[2])
axes[0].set_title('Logistic Regression')
axes[1].set_title('KNN')
axes[2].set_title('Naive Bayes')
plt.show()

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Naive Bayes'],
    'Accuracy': [acc_lr, acc_knn, acc_nb],
    'Confusion Matrix': [cm_lr, cm_knn, cm_nb],
    'Classification Report': [cr_lr, cr_knn, cr_nb]
})

results.to_csv('results.csv', index=False)

results = pd.read_csv('results.csv')
print(results)



#2)
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
url="https://www.indiatoday.in/world/story/iran-schoolgirls-poisoning-spread-panic-as-attack-spread-across-50-schools-2343047-2023-03-06"
r=requests.get(url)
soup=BeautifulSoup(r.content, 'html.parser')

text=""
for p in soup.find_all('p'):
    text += p.get_text()
nltk.download('stopwords')

tokens=word_tokenize(text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if not word in stop_words]
nltk.download('vader_lexicon')
nltk.download('punkt')
sid = SentimentIntensityAnalyzer()

labeled_tokens = []
for word in filtered_tokens:
    if sid.polarity_scores(word)['compound'] > 0:
        labeled_tokens.append((word, 'positive'))
    elif sid.polarity_scores(word)['compound'] < 0:
        labeled_tokens.append((word, 'negative'))
    else:
        labeled_tokens.append((word, 'neutral'))

train_size = int(0.8 * len(labeled_tokens))
train_data = labeled_tokens[:train_size]
test_data = labeled_tokens[train_size:]

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform([t[0] for t in train_data])
train_labels = [t[1] for t in train_data]

test_vectors = vectorizer.transform([t[0] for t in test_data])
test_labels = [t[1] for t in test_data]

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(train_vectors, train_labels)

predicted_labels = knn.predict(test_vectors)

print(classification_report(test_labels, predicted_labels, zero_division=0))
print(confusion_matrix(test_labels, predicted_labels))
text_vector = vectorizer.transform([word for word in filtered_tokens])
predicted_sentiment = knn.predict(text_vector)

positive_count=sum(1 for label in predicted_sentiment if label == 'positive')
negative_count=sum(1 for label in predicted_sentiment if label == 'negative')
neutral_count=sum(1 for label in predicted_sentiment if label == 'neutral')
sentiment_score=(positive_count - negative_count) / len(predicted_sentiment)
print("Sentiment score:", sentiment_score)
print("Positive count:", positive_count)
print("Negative count:", negative_count)
print("Neutral count:", neutral_count)

if sentiment_score > 0:
    print("The overall sentiment of the text is positive.")
elif sentiment_score < 0:
    print("The overall sentiment of the text is negative.")
else:
    print("The overall sentiment of the text is neutral.")
    
#this is not that accurate but most of the news articles it was able to predict right.


#3)credit card clustering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('CC GENERAL.csv')
data.dtypes
data.describe()

data=data.drop('CUST_ID',axis=1)

sns.heatmap(data.corr())
plt.show()

print(data.isnull().sum())

data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].median(), inplace=True)
data.dropna(subset=['MINIMUM_PAYMENTS'], inplace=True)


clustering_data = data[['CREDIT_LIMIT', 'PURCHASES', 'PURCHASES_FREQUENCY', 'PAYMENTS', 'TENURE','BALANCE']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(clustering_data)

data_scaled2=data_scaled3=data_scaled
data2=data3=data

#kmeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,n_init=5, init='k-means++', random_state=42)
kmeans.fit(data_scaled2)
pred = kmeans.fit_predict(data_scaled2)
data2['cluster'] = kmeans.labels_

sns.scatterplot(data=data2, x='PURCHASES', y='CASH_ADVANCE', hue='cluster', palette='bright')
plt.title('Clustered Credit Card Users')
plt.show()


print(data2.groupby('cluster').mean())


#hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
hc.fit(data_scaled3)
pred=hc.fit_predict(data_scaled3)
data3['cluster'] = hc.labels_


sns.scatterplot(data=data3, x='PURCHASES', y='CASH_ADVANCE', hue='cluster', palette='bright')
plt.title('Clustered Credit Card Users')
plt.show()

print(data3.groupby('cluster').mean())

