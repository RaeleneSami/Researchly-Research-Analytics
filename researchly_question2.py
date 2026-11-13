"""Researchly Questions 2.1 and 2.2. Run: python researchly_question2.py"""
from pathlib import Path
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
csv=ROOT/'cleaned_research_abstracts.csv'
if csv.exists():
    df=pd.read_csv(csv)
else:
    df=pd.read_json(ROOT/'research_abstracts.json')
    df=df.dropna(subset=['Abstract','Label']).drop_duplicates(subset=['Abstract']).copy()
    stemmer=PorterStemmer()
    stop_words=set(ENGLISH_STOP_WORDS)
    def clean_text(s):
        s=re.sub(r'http\S+|www\S+', ' ', str(s).lower())
        s=re.sub(r'[^a-z\s]', ' ', s)
        return ' '.join(stemmer.stem(w) for w in s.split() if w not in stop_words and len(w)>2)
    df['Cleaned_Abstract']=df['Abstract'].map(clean_text)

df=df.dropna(subset=['Cleaned_Abstract','Label']).copy()
df=df[df['Cleaned_Abstract'].str.strip().ne('')].copy()
X=df['Cleaned_Abstract'].astype(str)
y=df['Label'].astype(str)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

def pipe(clf, max_features=5000, ngram_range=(1,2)):
    return Pipeline([('tfidf', TfidfVectorizer(max_features=max_features,ngram_range=ngram_range,sublinear_tf=True)),('classifier',clf)])

models={
    'Baseline Multinomial NB':MultinomialNB(),
    'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),
    'Linear SVM':LinearSVC(class_weight='balanced',random_state=42),
    'Complement Naive Bayes':ComplementNB(),
    'SGD Classifier':SGDClassifier(loss='modified_huber',class_weight='balanced',random_state=42),
}
rows=[]
for name,clf in models.items():
    p=pipe(clf)
    cv_scores=cross_val_score(p,X_train,y_train,cv=cv,scoring='f1_macro',n_jobs=-1)
    p.fit(X_train,y_train)
    pred=p.predict(X_test)
    rows.append({'Model':name,'CV Macro F1':cv_scores.mean(),'Test Accuracy':accuracy_score(y_test,pred),'Test Macro F1':f1_score(y_test,pred,average='macro'),'AI Recall':recall_score(y_test,pred,pos_label='ai')})
results=pd.DataFrame(rows).sort_values('CV Macro F1',ascending=False)
print('\nQUESTION 2.1 – SAME HOLDOUT TEST SET FOR ALL MODELS')
print(results.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
results.to_csv(ROOT/'researchly_model_comparison.csv',index=False)

# Select using training cross-validation, NOT using holdout test scores.
selected=next(name for name in results['Model'] if name!='Baseline Multinomial NB')
print('\nSelected by training CV:',selected)

# One enhancement approach: hyperparameter optimisation of classifier regularisation
# and TF-IDF parameters by GridSearchCV on training data only.
params={
 'Logistic Regression': {'tfidf__max_features':[3000,5000,10000], 'tfidf__ngram_range':[(1,1),(1,2)],'classifier__C':[0.1,1,5]},
 'Linear SVM': {'tfidf__max_features':[3000,5000,10000], 'tfidf__ngram_range':[(1,1),(1,2)],'classifier__C':[0.1,1,5]},
 'Complement Naive Bayes': {'tfidf__max_features':[3000,5000,10000], 'tfidf__ngram_range':[(1,1),(1,2)],'classifier__alpha':[0.1,0.5,1]},
 'SGD Classifier': {'tfidf__max_features':[3000,5000,10000], 'tfidf__ngram_range':[(1,1),(1,2)],'classifier__alpha':[1e-5,1e-4,1e-3]},
}
search=GridSearchCV(pipe(models[selected]),params[selected],scoring='f1_macro',cv=cv,n_jobs=-1,refit=True)
search.fit(X_train,y_train)
best=search.best_estimator_
pred=best.predict(X_test)
initial_cv=float(results.set_index('Model').loc[selected,'CV Macro F1'])
print('\nQUESTION 2.2 – HYPERPARAMETER OPTIMISATION')
print('Previous CV Macro F1:',round(initial_cv,4))
print('Tuned CV Macro F1:',round(search.best_score_,4))
print('Improvement in CV Macro F1:',round(search.best_score_-initial_cv,4))
print('Optimal parameters:',search.best_params_)
print('Tuned holdout accuracy:',round(accuracy_score(y_test,pred),4))
print('Tuned holdout macro F1:',round(f1_score(y_test,pred,average='macro'),4))
print('Tuned AI recall:',round(recall_score(y_test,pred,pos_label='ai'),4))
print('\nClassification report:\n',classification_report(y_test,pred,zero_division=0))
print('Confusion matrix [ai,classical]:\n',confusion_matrix(y_test,pred,labels=['ai','classical']))
joblib.dump(best,ROOT/'researchly_tuned_model.joblib')
# Deploy the strongest original model, selected by training CV.
deployment=pipe(models[selected])
deployment.fit(X_train,y_train)
joblib.dump(deployment,ROOT/'researchly_best_model.joblib')
print('\nSaved original selected model for Question 2.3 and tuned model separately')
