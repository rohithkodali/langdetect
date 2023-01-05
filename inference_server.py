"""
Author : Rohith Gowtham
Date : 05/05/2021 17:47:52 PM

This is a web app to detect the language of a text using BPE and Logistic Regression. Built using FastAPI.
Load the model,labels,vectorizer and tokenizer from the pickle files
For comparision added fasttext based model for langauge prediction
"""

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import re
import numpy as np
import fasttext
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


app = FastAPI(title="Language Detection API", description="Languauge Detection API", version="1.0.0", openapi_url="/langdetect/openapi.json", docs_url="/langdetect/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

api = APIRouter()

#load the tokenizer, vectorizer,labels and model.pkl files for inference
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
clf = pickle.load(open('models/model.pkl', 'rb'))
labels = pickle.load(open('models/labels.pkl', 'rb'))

#loading fasttext model to show comparison with the classifier and other models available
fasttext_model = fasttext.load_model('lid.176.bin')

@api.post("/predict_framework")
async def predict_framework(text: str):
    '''
    using Facebook fasttext model to predict the language
    :param text: text to be classified
    :return: recognized language and confidence score in {language:lang, confidence:score} format
    '''
    try:
        predictions = fasttext_model.predict(text, k=1)  # top 1 matching languages
        return {"language": predictions[0][0].replace("__label__", ""), "confidence": predictions[1][0]}
    except:
        return {"error": "Malformed string", "Status_code": 400}

@api.post("/predict_customized")
async def predict(text: str):
    '''
    using the custom bilt classifier trained to predict the language
    :param text: text to be classified
    :return: recognized language and confidence score in {language:lang, confidence:score} format
    '''
    try:
        input_string=re.sub(r'[^\w\s\']','',text).lower()
        output = tokenizer.encode(input_string)
        features = vectorizer.transform([' '.join(output.tokens)])
        prediction = clf.predict_proba(features)
        print(labels[np.argmax(prediction)], "->", np.max(prediction))
        return {"language": labels[np.argmax(prediction)], "confidence": np.max(prediction)}
    except:
        return {"error": "Malformed string", "Status_code": 400}

app.include_router(api, prefix="/langdetect", responses={404: {"description": "Not found"}})

if __name__ == "__main__":
    nop=1   #increase the number if you want multiple workers
    uvicorn.run('inference_server:app', host="0.0.0.0", port=8888, workers=nop)
