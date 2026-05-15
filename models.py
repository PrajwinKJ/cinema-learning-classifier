from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import joblib
import tools

df=pd.read_csv('training data/training_3.csv')
df['language']=(df['language']=='ml').astype('Int64')
df=df.drop('tconst',axis=1)
null=df.isnull().sum()
null_drop=null[(null<=50)&(null>0)].index
df=df.dropna(subset=null_drop)
df=df.fillna('')
y=df.pop('language')
X=df
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

ct=ColumnTransformer([('title_vect',CountVectorizer(ngram_range=(1,3)),'originalTitle'),
                      ('genre_vect',CountVectorizer(tokenizer=tools.comma_split),'genres'),
                      ('directors_vect',CountVectorizer(),'directors'),
                      ('writers_vect',CountVectorizer(),'writers'),
                      ('crews_vect',CountVectorizer(),'crews')
                      ], remainder='drop')

pipeline=Pipeline([
    ('Vectorizer',ct),
    ('Classifier',RandomForestClassifier(random_state=42,class_weight='balanced',max_depth=50,n_jobs=3,min_samples_leaf=1,min_samples_split=7,n_estimators=100))
])

params={
    'Classifier__max_depth':[i for i in range(15,101,3)],
    'Classifier__n_estimators':[i for i in range(100,301,50)],
    'Classifier__min_samples_split':[2,5,7,10],
    'Classifier__min_samples_leaf':[1,2,4,6],
    'Classifier__max_features':['sqrt','log2']
}

search=RandomizedSearchCV(pipeline,param_distributions=params,n_iter=10,cv=10,scoring='f1',random_state=42,n_jobs=3)
search.fit(x_train,y_train)
model=search.best_estimator_

joblib.dump(model,'models/mal_model_v3.joblib')
print("Model Saved Successfuly!!")