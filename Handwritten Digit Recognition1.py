#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed


# In[ ]:


# setting a download path
import os

desktop_path = os.path.join(os.path.expanduser(r"C:\Users\Tamakloe Vivian A\Pictures\ML"))
desktop_path


# In[ ]:


# Load Dataset
mnist = fetch_openml('mnist_784', version=1)


# In[ ]:


# Define Features and Labels
X = mnist.data
y = mnist.target.astype(int)


# In[ ]:


# inspecting dataset 
print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)
print("Unique classes:", np.unique(y))


# In[ ]:


# Visualizing Sample Images
plt.figure(figsize=(8,6))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X.iloc[i].values.reshape(28,28), cmap='gray')
    plt.title(f"Label: {y.iloc[i]}")
    plt.axis('off')

plt.tight_layout()

plt.savefig(
    os.path.join(desktop_path, "Figure_4_1_Sample Images.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:


# checking class distibution
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution of MNIST Dataset")
plt.xlabel("Digit")
plt.ylabel("Frequency")

plt.savefig(
    os.path.join(desktop_path, "Figure_4_2_class_distribution.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:


# Normalizing Pixel Values
X = X.values
X = X / 255.0


# In[ ]:


# spliting for training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# In[ ]:


# importing linear regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)


# In[ ]:


# training
lr.fit(X_train, y_train)


# In[ ]:


# testing
y_pred_lr = lr.predict(X_test)


# In[ ]:


# Accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Print classification report
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr))


# In[ ]:


# subsampling SVM training data for speed
X_train_svm = X_train[:20000]
y_train_svm = y_train[:20000]


# In[ ]:


# importing SVM
from sklearn.svm import SVC


# In[ ]:


# initializing model
svm = SVC(
    kernel='rbf',
    C=5,
    probability=True,
    gamma='scale'
)


# In[ ]:


# training
svm.fit(X_train_svm, y_train_svm)


# In[ ]:


# testing
y_pred_svm = svm.predict(X_test)


# In[ ]:


# Accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Print classification report
print("SVM Classification Report")
print(classification_report(y_test, y_pred_svm))


# In[ ]:


# preparing data for CNN
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# normalizing again as safety precaution
X_train_cnn = X_train_cnn.astype("float32")
X_test_cnn = X_test_cnn.astype("float32")


# In[ ]:


# encoding labels
from tensorflow.keras.utils import to_categorical

y_train_cnn = to_categorical(y_train, 10)
y_test_cnn = to_categorical(y_test, 10)


# In[ ]:


# building CNN 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Input(shape=(28, 28, 1)),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# In[ ]:


# compiling model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


# training
history = model.fit(
    X_train_cnn,
    y_train_cnn,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)


# In[ ]:


# Accuracy
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cnn)
print("CNN Test Accuracy:", test_accuracy)


# In[ ]:


# testing
y_pred_cnn_probs = model.predict(X_test_cnn)

# converting to class labels
y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
y_true_cnn = np.argmax(y_test_cnn, axis=1)

# printing classification report
print("CNN Classification Report")
print(classification_report(y_true_cnn, y_pred_cnn))


# In[ ]:


# reshaping data for LSTM
X_train_rnn = X_train.reshape(-1, 28, 28)
X_test_rnn = X_test.reshape(-1, 28, 28)

X_train_rnn = X_train_rnn.astype("float32")
X_test_rnn = X_test_rnn.astype("float32")


# In[ ]:


# building LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

model_lstm = Sequential([
    Input(shape=(28, 28)),
    
    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])


# In[ ]:


# compiling
model_lstm.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


# training
history_lstm = model_lstm.fit(
    X_train_rnn,
    y_train_cnn,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)


# In[ ]:


# Accuracy
test_loss_lstm, test_accuracy_lstm = model_lstm.evaluate(X_test_rnn, y_test_cnn)
print("LSTM Test Accuracy:", test_accuracy_lstm)


# In[ ]:


# testing
y_pred_lstm_probs = model_lstm.predict(X_test_rnn)

y_pred_lstm = np.argmax(y_pred_lstm_probs, axis=1)

# printing classification report
print("LSTM Classification Report")
print(classification_report(y_true_cnn, y_pred_lstm))


# In[ ]:


# bulding hybrid architecture
model_hybrid = Sequential([
    Input(shape=(28,28,1)),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    # Converting feature maps into sequences
    Reshape((5*5, 64)),  # after pooling, shape becomes (5,5,64)
    
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])


# In[ ]:


# compiling
model_hybrid.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


# training
history_hybrid = model_hybrid.fit(
    X_train_cnn,
    y_train_cnn,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)


# In[ ]:


# hybrid reshape line
Reshape((5*5, 64))


# In[ ]:


# Accuracy
test_loss_hybrid, test_accuracy_hybrid = model_hybrid.evaluate(X_test_cnn, y_test_cnn)
print("Hybrid Test Accuracy:", test_accuracy_hybrid)


# In[ ]:


# testing
y_pred_hybrid_probs = model_hybrid.predict(X_test_cnn)

y_pred_hybrid = np.argmax(y_pred_hybrid_probs, axis=1)

# printing classification report
print("Hybrid CNN-LSTM Classification Report")
print(classification_report(y_true_cnn, y_pred_hybrid))


# In[ ]:


# reporting
report = classification_report(y_true_cnn, y_pred_cnn, output_dict=True)
pd.DataFrame(report).transpose()


# In[ ]:


# comparative accuracy chart

# Model names
models = [
    "Logistic Regression",
    "SVM (RBF)",
    "LSTM",
    "CNN",
    "CNN-LSTM Hybrid"
]

# computed values 
accuracies = [
    0.92,
    0.9759,
    0.9853,
    0.9896,
    0.9886
]

# Creating figure
plt.figure()

bars = plt.bar(models, accuracies)

# Adding value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.4f}",
        ha='center',
        va='bottom'
    )

# Axis labels and title
plt.xlabel("Models")
plt.ylabel("Test Accuracy")
plt.title("Comparative Model Accuracy on MNIST")

# Improving readability
plt.xticks(rotation=45)
plt.ylim(0.9, 1.0)

plt.tight_layout()

plt.savefig(
    os.path.join(desktop_path, "Figure_4_3_comparative accuracy.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:


# confusion matrix of best performing model

# generating predictions
y_pred_probs = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cnn, axis=1)

# computing confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ploting confusion matrix
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - CNN Model")

plt.savefig(
    os.path.join(desktop_path, "Figure_4_4_confusion_matrix.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:


# Training vs Validation Accuracy Curve
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy (CNN)")

plt.savefig(
    os.path.join(desktop_path, "Figure_4_5_training_vs_Validation_accuracy.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# In[ ]:


# Training vs Validation Loss Curve
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (CNN)")

plt.savefig(
    os.path.join(desktop_path, "Figure_4_6_training_vs_Validation_loss.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# In[ ]:


# Comparative Predicitve Performance table

# Creating dictionary to store results
results = {}

# Logistic Regression
results["Logistic Regression"] = {
    "Accuracy": accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr, average="weighted"),
    "Recall": recall_score(y_test, y_pred_lr, average="weighted"),
    "F1-Score": f1_score(y_test, y_pred_lr, average="weighted")
}

# SVM
results["SVM (RBF)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_svm),
    "Precision": precision_score(y_test, y_pred_svm, average="weighted"),
    "Recall": recall_score(y_test, y_pred_svm, average="weighted"),
    "F1-Score": f1_score(y_test, y_pred_svm, average="weighted")
}

# CNN
results["CNN"] = {
    "Accuracy": accuracy_score(y_true_cnn, y_pred_cnn),
    "Precision": precision_score(y_true_cnn, y_pred_cnn, average="weighted"),
    "Recall": recall_score(y_true_cnn, y_pred_cnn, average="weighted"),
    "F1-Score": f1_score(y_true_cnn, y_pred_cnn, average="weighted")
}

# LSTM
results["LSTM"] = {
    "Accuracy": accuracy_score(y_true_cnn, y_pred_lstm),
    "Precision": precision_score(y_true_cnn, y_pred_lstm, average="weighted"),
    "Recall": recall_score(y_true_cnn, y_pred_lstm, average="weighted"),
    "F1-Score": f1_score(y_true_cnn, y_pred_lstm, average="weighted")
}

# Hybrid
results["CNN-LSTM Hybrid"] = {
    "Accuracy": accuracy_score(y_true_cnn, y_pred_hybrid),
    "Precision": precision_score(y_true_cnn, y_pred_hybrid, average="weighted"),
    "Recall": recall_score(y_true_cnn, y_pred_hybrid, average="weighted"),
    "F1-Score": f1_score(y_true_cnn, y_pred_hybrid, average="weighted")
}

# converting to DataFrame
performance_table = pd.DataFrame(results).T

# Rounding values for clean presentation
performance_table = performance_table.round(4)

performance_table


# In[ ]:


performance_table.to_csv("comparative_performance_table.csv")


# In[ ]:


# Generate classification report
report = classification_report(y_test, y_pred_cnn, output_dict=True)

# Convert to dataframe
report_df = pd.DataFrame(report).transpose()

# Keep only class rows (0–9)
report_df = report_df.iloc[0:10]

print(report_df)


# In[ ]:


#plotting
plt.figure(figsize=(10,6))

sns.heatmap(
    report_df[['precision','recall','f1-score']],
    annot=True,
    cmap="viridis",
    fmt=".3f"
)

plt.title("Precision, Recall and F1-Score Heatmap for CNN Model")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Digit Classes")

plt.savefig(
    os.path.join(desktop_path, "Figure_4_7_Heatmap.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.show()


# In[ ]:


# Converting Labels to One-Hot Encoding
from sklearn.preprocessing import label_binarize

# Getting number of classes
n_classes = len(np.unique(y_test))

# Binarize labels
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))


# In[ ]:


# Logistic Regression Multi-Class ROC
from sklearn.metrics import roc_curve, auc

# Getting probability score
y_scores_lr = lr.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores_lr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Computing macro-average ROC
fpr["macro"], tpr["macro"], _ = roc_curve(
    y_test_bin.ravel(), y_scores_lr.ravel()
)
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plotting 
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label="Logistic Regression (Macro AUC = %0.4f)" % roc_auc["macro"])

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC - Logistic Regression")
plt.legend()

plt.savefig(
    os.path.join(desktop_path, "Figure_4_8_LogisticsRegression_ROC_Curve.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print("Logistic Regression Macro AUC:", roc_auc["macro"])


# In[ ]:


# SVM ROC

# getting probability scores
y_score_svm = svm.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

# computing ROC curve and AUC for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_svm[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# computing macro-average ROC curve

# aggregating all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# interpolating all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Averaging it
mean_tpr /= n_classes

roc_auc["macro"] = auc(all_fpr, mean_tpr)


# Plotting
plt.figure()
plt.plot(all_fpr, mean_tpr,
         label='SVM Macro-average ROC (AUC = %0.4f)' % roc_auc["macro"],
         linewidth=2)

# Plotting diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC Curve for SVM')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(desktop_path, "Figure_4_9_SVM_ROC_Curve.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print("SVM Macro-average AUC:", roc_auc["macro"])


# In[ ]:


# CNN ROC

# getting probability predictions
y_pred_cnn_probs = model.predict(X_test_cnn)

# computing ROC curve and AUC 
fpr["macro"], tpr["macro"], _ = roc_curve(
    y_test_bin.ravel(), y_pred_cnn_probs.ravel()
)

roc_auc_macro = auc(fpr["macro"], tpr["macro"])

# Plotting
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label="CNN (Macro AUC = %0.4f)" % roc_auc_macro)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC - CNN")
plt.legend()

plt.savefig(
    os.path.join(desktop_path, "Figure_4_10_CNN_ROC_Curve.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print("CNN Macro AUC:", roc_auc_macro)


# In[ ]:


# LSTM ROC

#getting probabilities
y_pred_lstm_probs = model_lstm.predict(X_test_rnn)

# computing ROC curve and AUC
fpr["macro"], tpr["macro"], _ = roc_curve(
    y_test_bin.ravel(), y_pred_lstm_probs.ravel()
)

roc_auc_macro = auc(fpr["macro"], tpr["macro"])

# Plotting
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label="LSTM (Macro AUC = %0.4f)" % roc_auc_macro)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC - LSTM")
plt.legend()

plt.savefig(
    os.path.join(desktop_path, "Figure_4_11_LSTM_ROC_Curve.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print("LSTM Macro AUC:", roc_auc_macro)


# In[ ]:


# hybrid ROC

# getting probabilities
y_pred_hybrid_probs = model_hybrid.predict(X_test_cnn)

# computing ROC curve and AUC
fpr["macro"], tpr["macro"], _ = roc_curve(
    y_test_bin.ravel(), y_pred_hybrid_probs.ravel()
)

roc_auc_macro = auc(fpr["macro"], tpr["macro"])

# Plotting
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label="Hybrid (Macro AUC = %0.4f)" % roc_auc_macro)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC - Hybrid")
plt.legend()

plt.savefig(
    os.path.join(desktop_path, "Figure_4_12_Hybrid_ROC_Curve.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print("Hybrid Macro AUC:", roc_auc_macro)


# In[ ]:


# Pairwise McNemar test
from scipy.stats import chi2

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    # Correctness masks
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    # Discordant pairs
    b01 = np.sum((a_correct) & (~b_correct))
    b10 = np.sum((~a_correct) & (b_correct))

    # McNemar chi-square statistic
    chi_square = (abs(b01 - b10) - 1)**2 / (b01 + b10)
    p_value = 1 - chi2.cdf(chi_square, df=1)

    return chi_square, p_value

# CNN vs Hybrid
chi2_stat, p = mcnemar_test(y_true, y_pred_cnn, y_pred_hybrid)
print("Chi-square:", chi2_stat)
print("p-value:", p)


# In[ ]:


#Bootstrap test for difference in accuracy (confidence interval)

def bootstrap_acc_diff(y_true, y_pred_a, y_pred_b, n_bootstraps=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)  # sample with replacement
        acc_a = np.mean(y_pred_a[idx] == y_true[idx])
        acc_b = np.mean(y_pred_b[idx] == y_true[idx])
        diffs[i] = acc_a - acc_b

    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    return mean_diff, (lower, upper)

# comparing CNN to Hybrid
mean_diff, ci = bootstrap_acc_diff(y_true_cnn, y_pred_cnn, y_pred_hybrid, n_bootstraps=2000)
print("Mean accuracy diff (CNN - Hybrid):", mean_diff, "95% CI:", ci)


# In[ ]:


# computing parameter counts and wall-clock training time

#parameter counting for Keras models
print("CNN params:", model.count_params())
print("LSTM params:", model_lstm.count_params())
print("Hybrid params:", model_hybrid.count_params())

# Measuring wall-clock time for a training run (example)
import time
t0 = time.time()

# model_cnn.fit(X_train_cnn, y_train_cnn, epochs=1, batch_size=64)
t1 = time.time()
print("Elapsed seconds:", t1 - t0)

