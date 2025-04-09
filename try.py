import lmstudio as lms

model = lms.llm('meta-llama-3.1-8b-instruct')

embed_model = lms.embedding_model('text-embedding-all-minilm-l6-v2-embedding')

text = """
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features.
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)
"""

embeddings = embed_model.embed(text)
print(embeddings)
