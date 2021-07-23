from fastai.vision.all import *
# from fastai.vision.core import PILImage
# from fastai.learner import load_learner
import streamlit as st
import requests
import os

@st.cache(allow_output_mutation=True)
def get_learner(file_name='export.pkl'):
    learn = load_learner(file_name)
    return learn


def download_files(URL):
    with open("export.pkl", "wb") as model:
        r = requests.get(URL)
        model.write(r.content)
    try:
        assert(os.path.getsize("export.pkl") > 15000)
        st.success('Model successfully downloaded!')
    except:
        st.warning("There is something wrong!")


def write():
    pl = st.empty()
    pl.markdown('''
    <html>
    <body style="background-color:#216D6D;">
    <h1 align="center" style="color:white;">Image Classifier</h1>
    </body>
    </html>
    ''',unsafe_allow_html=True)
    if (not os.path.isfile('export.pkl') or os.path.getsize("export.pkl") < 15000):
        ph = st.empty()
        ph2 = st.empty()
        ph3 = st.empty()
        ph.warning('Please download the model file')
        URL = ph3.text_input('Please input the URL','')
        if ph2.button('Download'):
            ph.empty()
            ph3.empty()
            ph2.text('Downloading...')
            try:
                download_files(URL)
                ph2.text('Download completed')
                st.button("Next Stage")
            except Exception as e:
                st.error('Not a correct URL!')
                print(str(e))

    else:
        st.success("Model already downloaded")

        img_data = st.file_uploader('Please upload your image',type=['jpg','jpeg','png'])
        if img_data == None:
                st.warning('Checking data...')
        else:
            st.image(img_data,use_column_width=True)
            check = st.button('Predict')
            if check:
                file_name = 'export.pkl'
                learn = get_learner(file_name)
                img = PILImage.create(img_data)
                result = learn.predict(img)
                pred,pred_idx,probs = result
                st.write('Result: '+pred.capitalize())
                st.write('Probablitiy: '+str(probs))


if __name__ == "__main__":
    write()

    