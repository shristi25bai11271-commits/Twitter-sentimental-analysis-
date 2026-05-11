import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample tweet dataset
data = {
    "tweet": [
        "I love this movie",
        "This is amazing",
        "I hate this app",Terminal: Select Default Profile
        "Very bad experience",
        "The product is okay",
        "Nothing special",
        "I am very happy",
        "This is terrible",
        "Average performance",
        "Best day ever"
    ],

    "sentiment": [
        "positive",
        "positive",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive"
    ]
}

# Convert into dataframe
df = pd.DataFrame(data)

# Input and output
X = df["tweet"]
y = df["sentiment"]

# Convert words into numbers
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# User input
user_tweet = input("Enter a tweet: ")

# Convert input
user_data = cv.transform([user_tweet])

# Predict sentiment
prediction = model.predict(user_data)

print("Sentiment:", prediction[0])