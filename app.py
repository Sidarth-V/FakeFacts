from flask import Flask, render_template
import pandas as pd
import string
import numpy as np 
import nltk
nltk.download('stopwords')
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from flask import Flask, render_template

f = pd.read_csv("./Fake.csv")
t = pd.read_csv("./True.csv")
f['target'] = 'fake'
t['target'] = 'true'
dt = pd.concat([f, t]).reset_index(drop = True)

dt = shuffle(dt)
dt = dt.reset_index(drop=True)
dt.drop(["date"],axis=1,inplace=True)
dt.drop(["title"],axis=1,inplace=True)
dt['text'] = dt['text'].apply(lambda x: x.lower())

def punc_r(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
dt['text'] = dt['text'].apply(punc_r)

stop = stopwords.words('english')

dt['text'] = dt['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
X_train,X_test,y_train,y_test = train_test_split(dt['text'], dt.target, test_size=0.2, random_state=40)

pie = Pipeline([('vect', CountVectorizer()), 
                ('tfidf', TfidfTransformer()), 
                ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 30, 
                                           random_state=40))])
model = pie.fit(X_train, y_train)
prediction = model.predict(X_test)

x = "Run this form"
#x = input("ENTER THE NEWS ARTICLE HERE : ")
x=[x,]
pred = model.predict(x)
print('THE GIVEN NEWS ARTICLE IS ' + str(pred))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    projectpath = request.form['w-node-3d38e3e1fc94-544794fd']
    pred = model.predict(str(projectpath))
    return str(pred)

if __name__ == '__main__':
    app.run()