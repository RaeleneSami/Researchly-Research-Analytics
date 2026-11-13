import pandas as pd, re, json, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from nltk.stem import SnowballStemmer
p='/mnt/data/research_abstracts.json'
df=pd.read_json(p)
print('RAW SHAPE',df.shape)
print('COLUMNS',df.columns.tolist())
print('NULLS',df.isna().sum().to_dict())
print('DUP_ROWS',df.duplicated().sum(),'DUP_ABSTRACT',df['Abstract'].duplicated().sum())
print('LABEL COUNTS',df['Label'].value_counts(dropna=False).to_dict())
print('EMPTY ABSTRACT',df['Abstract'].fillna('').astype(str).str.strip().eq('').sum())
print('FIRST',str(df['Abstract'].iloc[0])[:300])
df=df.dropna(subset=['Abstract','Label']).drop_duplicates(subset=['Abstract']).copy()
stemmer=SnowballStemmer('english')
def denoise(s):
 s=str(s).lower()
 s=re.sub(r'https?://\S+|www\.\S+', ' ', s)
 s=re.sub(r'[^a-z\s]', ' ', s)
 tokens=re.findall(r'\b[a-z]{3,}\b',s)
 return ' '.join(stemmer.stem(w) for w in tokens if w not in ENGLISH_STOP_WORDS)
df['cleaned_abstract']=df['Abstract'].map(denoise)
df=df[df['cleaned_abstract'].str.strip().ne('')].copy()
print('CLEAN SHAPE',df.shape)
print('CLEAN LABEL COUNTS',df['Label'].value_counts().to_dict())
print('CLEAN EXAMPLE',df['cleaned_abstract'].iloc[0][:280])
X_train,X_test,y_train,y_test=train_test_split(df['cleaned_abstract'],df['Label'],test_size=.2,random_state=42,stratify=df['Label'])
model=Pipeline([('tfidf',TfidfVectorizer(max_features=5000,ngram_range=(1,2),sublinear_tf=True)),('clf',MultinomialNB())])
model.fit(X_train,y_train)
yptrain=model.predict(X_train); yptest=model.predict(X_test)
print('SPLIT',len(X_train),len(X_test))
print('TRAIN ACC',accuracy_score(y_train,yptrain),'TEST ACC',accuracy_score(y_test,yptest),'GAP',accuracy_score(y_train,yptrain)-accuracy_score(y_test,yptest))
print('REPORT\n',classification_report(y_test,yptest,digits=4))
print('CM LABELS',model.classes_.tolist(),'CM',confusion_matrix(y_test,yptest,labels=model.classes_).tolist())
cv=cross_val_score(model,X_train,y_train,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42),scoring='f1_macro')
print('CV',cv.tolist(),cv.mean(),cv.std())
print('FEATURES',len(model['tfidf'].get_feature_names_out()))
print('TRAIN LABEL',y_train.value_counts().to_dict(),'TEST LABEL',y_test.value_counts().to_dict())
with open('/mnt/data/researchly_results.json','w') as f:json.dump(dict(rawshape=list(pd.read_json(p).shape),nulls=pd.read_json(p).isna().sum().to_dict(),rawlabels=pd.read_json(p)['Label'].value_counts().to_dict(),cleanedshape=list(df.shape),cleanedlabels=df['Label'].value_counts().to_dict(),train=len(X_train),test=len(X_test),trainacc=accuracy_score(y_train,yptrain),testacc=accuracy_score(y_test,yptest),gap=accuracy_score(y_train,yptrain)-accuracy_score(y_test,yptest),report=classification_report(y_test,yptest,output_dict=True),labels=model.classes_.tolist(),cm=confusion_matrix(y_test,yptest,labels=model.classes_).tolist(),cv=cv.tolist(),cvmean=cv.mean(),cvstd=cv.std(),nfeatures=len(model['tfidf'].get_feature_names_out()),cleaned_example=df['cleaned_abstract'].iloc[0]),f,indent=2)
