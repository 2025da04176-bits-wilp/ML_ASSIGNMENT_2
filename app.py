import streamlit as st, pandas as pd, joblib, os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title='Digits Classification', layout='wide')
st.title('Handwritten Digits Classification (0-9)')
st.write('Upload test_data.csv and evaluate any trained model.')

uploaded=st.file_uploader('Upload test_data.csv',type='csv')
choice=st.selectbox('Select Model',['Logistic Regression','Decision Tree','kNN','Naive Bayes','Random Forest'])
files={'Logistic Regression':'Logistic_Regression.pkl','Decision Tree':'Decision_Tree.pkl','kNN':'kNN.pkl','Naive Bayes':'Naive_Bayes.pkl','Random Forest':'Random_Forest.pkl'}

if uploaded:
    df=pd.read_csv(uploaded)
    X=df.drop('target',axis=1)
    y=df['target']
    model=joblib.load(os.path.join('model',files[choice]))
    pred=model.predict(X)
    prob=model.predict_proba(X)
    auc=roc_auc_score(label_binarize(y,classes=range(10)),prob,multi_class='ovr',average='weighted')
    c1,c2,c3=st.columns(3)
    c1.metric('Accuracy',f'{accuracy_score(y,pred):.4f}')
    c2.metric('AUC',f'{auc:.4f}')
    c3.metric('F1',f'{f1_score(y,pred,average="weighted"):.4f}')
    st.write({'Precision':precision_score(y,pred,average='weighted'),'Recall':recall_score(y,pred,average='weighted'),'MCC':matthews_corrcoef(y,pred)})
    st.subheader('Confusion Matrix')
    st.dataframe(pd.DataFrame(confusion_matrix(y,pred)))
    st.subheader('Classification Report')
    st.text(classification_report(y,pred))
else:
    st.info('Upload the provided test_data.csv to begin.')
