✅ 1. Ethical Considerations
💡 Biases in the MNIST Model (Digit Classification)
While MNIST is relatively balanced and objective (digits 0–9), some ethical concerns include:

Underrepresentation of certain writing styles (e.g., left-handed digits, non-Western handwriting)

Overfitting to neat, clearly written digits — might fail on real-world digits written by children or elderly people

🔧 Mitigation using TensorFlow Fairness Indicators:

Use Fairness Indicators to evaluate model performance across slices (e.g., handwriting from different age groups or cultural sources)

Monitor metrics like accuracy disparity, false positives, false negatives across subgroups

💡 Biases in the Amazon Reviews Model (spaCy + rule-based sentiment)
Rule-based methods can be oversimplified and biased towards:

Positive words in sarcastic context: “Great job breaking it, hero.”

Neglecting cultural language or slang (e.g., “bad” as a compliment)

🔧 Mitigation using spaCy’s Custom Rules:

Expand positive/negative word lists with contextual synonyms

Use custom pipelines with spaCy to account for sarcasm or domain-specific words

Combine with ML-based sentiment classifiers for better generalization



✅ 2. Troubleshooting Challenge: Fixing Buggy TensorFlow Code
❌ Example Buggy Code (Common Issues)
python
Copy
Edit
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
🔍 Problem:
Using mean_squared_error for classification (not appropriate)

Might cause shape mismatch if y_train is one-hot encoded and predictions are raw logits

✅ Fixed Code:
python
Copy
Edit
from tensorflow.keras.utils import to_categorical

# Correct loss for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ensure labels are one-hot encoded
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Proceed with training
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
