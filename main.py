# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle

# Load your dataset (make sure the CSV file is in the same folder or provide the correct path)
df = pd.read_csv('legal_clauses.csv')

# Shuffle the dataset for randomness
df = shuffle(df, random_state=42)

# Split data into input (X) and labels (y)
X = df['clause_text']
y = df['label']

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Convert text data into TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

# Sample predictions
sample_text = [
    "The borrower agrees to pay the full amount within 30 days.",
    "The contract shall be governed by the laws of India.",
    "All disputes shall be resolved through arbitration."
]
sample_features = tfidf.transform(sample_text)
sample_predictions = clf.predict(sample_features)

print("\nSample Predictions:")
for text, prediction in zip(sample_text, sample_predictions):
    print(f"Clause: \"{text}\" --> Predicted Label: {prediction}")
