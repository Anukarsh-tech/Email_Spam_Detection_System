import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()


def transform_text(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)

    y = []
    for i in Message:
        if i.isalnum():
            y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
        y.append(ps.stem(i))

    return " ".join(Message)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Spam Detection System")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

   # 1.Preprocess
   transform_text(input_sms)
   # 2. Vectorize
   vector_input = tfidf.transform([input_sms])
   # 3. Predict
   result = model.predict(vector_input)[0]
   # 4. Display
   if result == 1:
       st.header("Spam")
   else:
       st.header("Not spam")
