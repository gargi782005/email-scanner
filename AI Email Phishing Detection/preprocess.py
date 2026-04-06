import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("data/emails.csv", encoding="latin1")

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"\d+","",text)
    text = re.sub(r"[^\w\s]","",text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

df["message"] = df["message"].apply(clean)

df.to_csv("data/cleaned.csv",index=False)

print("Preprocessing Done Successfully ")