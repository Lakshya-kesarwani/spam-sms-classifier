import streamlit as st
import pickle
import string
import sklearn
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    res = list(filter( str.isalnum , text))
    text = res[:]
    res.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            res.append(i)
    res = [ps.stem(word) for word in res]
    return ' '.join(res)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter The message")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: #FAFAFA;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    Made with ❤️ by Lakshya | July, 2024
</div>
""", unsafe_allow_html=True)
