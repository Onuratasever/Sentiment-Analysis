import numpy as np
import pandas as pd
import re # regular expression
import nltk # natural language tool kit
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

comments = []
with open('Restaurant_Reviews.csv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        parts = line.rsplit(',', 1)
        if len(parts) == 2:
            comments.append(parts)

comments = pd.DataFrame(comments, columns=['Review', 'Liked'])

#print(comments.head())

#Preprocessing
nltk.download('stopwords')

ps = PorterStemmer() # Find roots of words

clean_comments = []
for i in range(1000):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    clean_comments.append(comment)
    
#Feautre Extraction
#Bag of Words (BOW)
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(clean_comments).toarray() # bağımsız değişken
y = comments.iloc[:,1].values # bağımlı değişken

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)