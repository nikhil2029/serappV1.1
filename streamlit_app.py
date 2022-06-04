import streamlit as st
import joblib
import librosa
import numpy as np
from keras.models import load_model

emotions={0:'angry',1:'fear',2:'happy',3:'sad'}
#feature_extraction
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

st.title("SERAPP V1.1")
file_bytes = st.file_uploader("Upload audio file", type=("mp3", "wav","m4a"))
st.audio(file_bytes)
alg=st.selectbox("Select Algorithm",("CNN","RandomForest"))
st.write(alg)
if(alg=='CNN'):
    st.write('Validation Accuraxy: 73.70%')
else:
    st.write('Validation Accuraxy: 73.9%')
if st.button('Predict'):
    if(file_bytes):
        data, sample_rate = librosa.load(file_bytes, duration=2.5, offset=0.6)
        res1 = extract_features(data, sample_rate)
        result = np.array(res1)
        test=np.expand_dims(result,axis=0)
        if(alg=='CNN'):
            model=load_model('CNN1DFIN.h5')
            res=model.predict(test)
            out=0
            st.write(res)
            for i in range(len(res[0])):
                if(res[0][i]==1.0):
                    out=i
                    break
        elif(alg=='RandomForest'):
            model1=joblib.load("./RFFIN.joblib")
            out=model1.predict(test)
            out=int(out[0])
    else:
        st.write("*please upload an audio file*")
    

else:

    st.write('click the button to predict')



