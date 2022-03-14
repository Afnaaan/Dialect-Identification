import imp
from fastapi import FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
import numpy as np
from scipy import stats


MAX_NUM_WORDS = 1000
MAX_SEQUENCE_LENGTH = 285
DIALECTS = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM', 'PL', 'QA', 'SA', 'SD', 'SY', 'TN', 'YE']

# loading
loaded_model = load_model('DL_model.h5')
tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return({"message": "It works!"})

@app.post("/detect_dialect/{text}")
async def detect_dialect(text):
    text = finalpreprocess(str(text))
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = loaded_model.predict(X)
    percentage = np.argmax(y, axis=1)
    idx = stats.mode(list(percentage))[0][0]
    return DIALECTS[idx]


def remove_tags_urls(string):
    return ' '.join(re.sub("(@[A-Za-z0-9-_]+)|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)","", string).split())

def remove_punc(string):
    return ' '.join(re.sub("[\.\,\!\?\:\;\-\=\"\'\_]","", string).split())

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def finalpreprocess(string_):
    string_ = remove_tags_urls(str(string_))
    string_ = remove_punc(string_)
    return remove_emojis(string_)