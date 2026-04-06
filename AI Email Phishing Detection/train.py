import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load your dataset
# Ensure your CSV has columns named 'message' and 'label'
df = pd.read_csv("data/cleaned.csv")

# Basic cleaning
df["message"] = df["message"].fillna("")
df.dropna(subset=["label"], inplace=True)

# Convert text to numbers
my_vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X = my_vectorizer.fit_transform(df["message"])
y = df["label"]

# Train the Classifier
my_model = LinearSVC()
my_model.fit(X, y)

# Save to the 'model' folder
pickle.dump(my_model, open("model/model.pkl", "wb"))
pickle.dump(my_vectorizer, open("model/vectorizer.pkl", "wb"))

print("Successfully trained and saved the model!")