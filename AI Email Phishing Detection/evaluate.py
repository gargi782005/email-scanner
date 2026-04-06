import pickle
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score

model = pickle.load(open("model/model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

df = pd.read_csv("data/cleaned.csv")

df["message"] = df["message"].fillna("")
df.dropna(subset=["label"], inplace=True)

X = vectorizer.transform(df["message"])
y = df["label"]

pred = model.predict(X)

print("Accuracy:",accuracy_score(y,pred))
print(classification_report(y,pred))