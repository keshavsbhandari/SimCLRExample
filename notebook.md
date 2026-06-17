## Step 1: Problem statement

### Project title

**Self-Supervised Fraud Detection Using SimCLR-Style Contrastive Learning**

---

### Problem statement

Fraud detection is a highly imbalanced binary classification problem where the goal is to identify fraudulent transactions among mostly legitimate transactions. In real-world transaction data, fraud labels can be limited, expensive, delayed, or noisy, while large amounts of unlabeled transaction data are usually available.

The goal of this project is to use a **SimCLR-style self-supervised contrastive learning approach** to learn useful transaction representations before training a supervised fraud classifier. Instead of training only from fraud labels, the model first learns from the structure of the transaction data itself by creating two augmented views of the same transaction and pulling their embeddings closer while pushing embeddings of different transactions apart.

After self-supervised pretraining, the learned encoder will be evaluated on a downstream fraud detection task using appropriate fraud detection metrics such as **PR-AUC, ROC-AUC, precision, recall, F1-score, and confusion matrix**.

---

### Why this problem matters

Fraud detection is a practical machine learning problem because false negatives can allow fraudulent activity to pass undetected, while false positives can incorrectly block legitimate users. The IEEE-CIS Fraud Detection dataset is designed around predicting whether an online transaction is fraudulent, with transaction and identity data joined by `TransactionID`. ([Kaggle][1])

This makes the problem suitable for testing whether self-supervised representation learning can improve downstream fraud classification, especially when labels are limited or class imbalance is severe.

---

### Why SimCLR is relevant here

SimCLR is a self-supervised contrastive learning framework that learns representations by comparing different augmented views of the same sample against other samples in the batch. The original SimCLR work showed that data augmentations, a nonlinear projection head, larger batch sizes, and contrastive learning can produce useful representations without using labels during pretraining. ([arXiv][2])

In this project, we adapt the same idea from images to tabular transaction data:

```text
Transaction x
   ↓
Augmentation 1 → view x1
Augmentation 2 → view x2
   ↓
Encoder + projection head
   ↓
Contrastive loss
```

The model learns that:

```text
two augmented views of the same transaction = positive pair
views from different transactions = negative pairs
```

---

### Main research question

**Can SimCLR-style self-supervised pretraining learn useful transaction embeddings that improve fraud detection compared with a supervised model trained from scratch?**

---

### Concrete project objective

Build and evaluate a fraud detection pipeline with the following goal:

```text
Use self-supervised contrastive pretraining on transaction data,
then fine-tune a classifier to predict fraudulent transactions.
```

The final model should answer:

```text
Given a transaction, what is the probability that it is fraudulent?
```

---

### Expected comparison

To make the project meaningful, compare at least these two models:

| Model                       | Description                                                                     |
| --------------------------- | ------------------------------------------------------------------------------- |
| **Baseline MLP**            | Train a neural network directly using fraud labels                              |
| **SimCLR + MLP classifier** | Pretrain encoder using contrastive learning, then fine-tune for fraud detection |

Optional stronger baselines:

| Model               | Why useful                        |
| ------------------- | --------------------------------- |
| Logistic Regression | Simple classical baseline         |
| Random Forest       | Nonlinear traditional ML baseline |
| XGBoost / LightGBM  | Strong tabular ML baseline        |

---

### Success criteria

The SimCLR-based model is useful if it improves one or more important fraud detection metrics, especially:

```text
PR-AUC
recall at high precision
F1-score
ROC-AUC
```

For fraud detection, **accuracy should not be the primary metric**, because fraud datasets are usually imbalanced. A model can get high accuracy by predicting almost everything as non-fraud, while still missing actual fraud cases.

---

### Resources

Use these as references for the problem statement:

1. **IEEE-CIS Fraud Detection dataset** - transaction and identity data for online fraud prediction. ([Kaggle][1])
2. **SimCLR paper** - original contrastive self-supervised learning framework. ([arXiv][2])
3. **Google Research SimCLR repository** - implementation/reference materials for SimCLR and SimCLRv2. ([github.com][3])

[1]: https://www.kaggle.com/competitions/ieee-fraud-detection/data?utm_source=chatgpt.com "IEEE-CIS Fraud Detection data"
[2]: https://arxiv.org/abs/2002.05709?utm_source=chatgpt.com "A Simple Framework for Contrastive Learning of Visual Representations"
[3]: https://github.com/google-research/simclr?utm_source=chatgpt.com "google-research/simclr: SimCLRv2 - Big Self-Supervised ..."

## Step 2: Dataset description

### Dataset

We will use the **PaySim Synthetic Financial Dataset** from Kaggle.

PaySim is a synthetic mobile-money transaction dataset generated to simulate financial transactions. It contains transaction-level records such as transaction type, amount, sender balance, receiver balance, and fraud labels.

Dataset link:
[https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

### Prediction task

The goal is binary fraud detection:

```text
Given a transaction, predict whether it is fraudulent.
```

Target column:

```text
isFraud
```

Label meaning:

```text
0 = legitimate transaction
1 = fraudulent transaction
```

---

### Main features

| Column           | Meaning                                               |
| ---------------- | ----------------------------------------------------- |
| `step`           | Time step of the transaction                          |
| `type`           | Transaction type, such as CASH_OUT, TRANSFER, PAYMENT |
| `amount`         | Transaction amount                                    |
| `oldbalanceOrg`  | Sender balance before transaction                     |
| `newbalanceOrig` | Sender balance after transaction                      |
| `oldbalanceDest` | Receiver balance before transaction                   |
| `newbalanceDest` | Receiver balance after transaction                    |
| `nameOrig`       | Sender account ID                                     |
| `nameDest`       | Receiver account ID                                   |
| `isFraud`        | Fraud label                                           |
| `isFlaggedFraud` | Rule-based fraud flag                                 |

---

### How it fits SimCLR

For SimCLR pretraining, we ignore `isFraud` and treat each transaction as an unlabeled sample.

```text
transaction x
   ↓
augmentation 1 → view x1
augmentation 2 → view x2
   ↓
contrastive learning
```

The model learns:

```text
two augmented views of the same transaction → close embeddings
views from different transactions → farther embeddings
```

After pretraining, we use `isFraud` to fine-tune and evaluate a fraud classifier.

---

### Important notes

PaySim is **synthetic**, so results should be presented as a controlled experiment rather than real-world banking performance.

The dataset is highly imbalanced, so accuracy should not be the main metric. We should focus on:

```text
PR-AUC
ROC-AUC
precision
recall
F1-score
confusion matrix
```

---

## Step 3: Preprocessing

Now we will preprocess the **PaySim** dataset instead of IEEE-CIS.

PaySim is available as a regular Kaggle dataset, so it avoids the expired IEEE-CIS competition access issue. The dataset includes mobile-money transaction features and a fraud label `isFraud`. The Kaggle dataset page also notes an important leakage warning: for fraud detection, balance columns such as `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest` should not be used because detected fraud transactions are cancelled. ([Kaggle][1])

---

### 3.1 Download and load PaySim

```python
!pip install kagglehub -q

import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("ealaxi/paysim1")

print("Dataset path:", path)
print("Files:", os.listdir(path))

csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]

df = pd.read_csv(os.path.join(path, csv_file))

print("Shape:", df.shape)
display(df.head())
```

`kagglehub.dataset_download()` downloads Kaggle datasets into a local path that you can then load with Pandas. ([GitHub][2])

---

### 3.2 Quick dataset check

```python
print(df.columns)

print("\nClass counts:")
print(df["isFraud"].value_counts())

print("\nClass ratio:")
print(df["isFraud"].value_counts(normalize=True))
```

This tells us how imbalanced the fraud label is.

---

### 3.3 Select target and remove leakage columns

Target:

```python
target_col = "isFraud"
```

For a clean fraud detection experiment, remove:

```text
isFraud           -> target label
isFlaggedFraud    -> rule-based fraud flag, possible shortcut
nameOrig          -> sender ID
nameDest          -> receiver ID
oldbalanceOrg     -> leakage warning from dataset page
newbalanceOrig    -> leakage warning from dataset page
oldbalanceDest    -> leakage warning from dataset page
newbalanceDest    -> leakage warning from dataset page
```

Code:

```python
target_col = "isFraud"

drop_cols = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

y = df[target_col].astype(int)
X = df.drop(columns=drop_cols)

print("X shape:", X.shape)
print("y shape:", y.shape)
display(X.head())
```

After this, the main remaining features are usually:

```text
step
type
amount
```

This is intentionally conservative to avoid leakage. Later, you can run an ablation where you add balance features back and compare results, but report it clearly as a separate experiment.

---

### 3.4 Train, validation, test split

Use stratified splitting because fraud is rare.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape, "fraud ratio:", y_train.mean())
print("Val:  ", X_val.shape, "fraud ratio:", y_val.mean())
print("Test: ", X_test.shape, "fraud ratio:", y_test.mean())
```

---

### 3.5 Identify numeric and categorical columns

```python
numeric_cols = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
```

For PaySim, you will likely get:

```text
numeric: step, amount
categorical: type
```

---

### 3.6 Build preprocessing pipeline

Numeric features:

```text
median imputation
standard scaling
```

Categorical features:

```text
missing-value imputation
one-hot encoding
```

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ]
)
```

---

### 3.7 Fit only on training data

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("Processed train:", X_train_processed.shape)
print("Processed val:  ", X_val_processed.shape)
print("Processed test: ", X_test_processed.shape)
```

Fit only on training data to avoid data leakage.

---

### 3.8 Convert to PyTorch tensors

```python
import numpy as np
import torch

X_train_tensor = torch.tensor(X_train_processed.astype(np.float32))
X_val_tensor = torch.tensor(X_val_processed.astype(np.float32))
X_test_tensor = torch.tensor(X_test_processed.astype(np.float32))

y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

print(X_train_tensor.shape, y_train_tensor.shape)
print(X_val_tensor.shape, y_val_tensor.shape)
print(X_test_tensor.shape, y_test_tensor.shape)
```

---

### 3.9 Save preprocessing artifacts

```python
import joblib

joblib.dump(preprocessor, "preprocessor.joblib")

feature_info = {
    "target_col": target_col,
    "drop_cols": drop_cols,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "input_dim": X_train_tensor.shape[1]
}

joblib.dump(feature_info, "feature_info.joblib")

print("Saved preprocessor.joblib and feature_info.joblib")
```

---

### 3.10 Final output of preprocessing

At the end of this step, you should have:

```text
X_train_tensor
X_val_tensor
X_test_tensor

y_train_tensor
y_val_tensor
y_test_tensor

preprocessor.joblib
feature_info.joblib
```

These are ready for:

```text
Step 4: SimCLR tabular augmentations
```

---

[1]: https://www.kaggle.com/datasets/ealaxi/paysim1/versions/2?resource=download&utm_source=chatgpt.com "Synthetic Financial Datasets For Fraud Detection"
[2]: https://github.com/Kaggle/kagglehub?utm_source=chatgpt.com "Kaggle/kagglehub: Python library to access ..."

### 3.11 Create PyTorch Dataset classes

We need two dataset classes:

| Dataset                | Used for                       | Returns  |
| :--------------------- | :----------------------------- | :------- |
| `SimCLRTabularDataset` | self-supervised pretraining    | `x1, x2` |
| `FraudDataset`         | supervised training/evaluation | `x, y`   |

```python
from torch.utils.data import Dataset, DataLoader
import torch

class SimCLRTabularDataset(Dataset):
    def __init__(self, X, feature_dropout=0.15, noise_std=0.05):
        self.X = X
        self.feature_dropout = feature_dropout
        self.noise_std = noise_std

    def augment(self, x):
        x_aug = x.clone()

        # Random feature dropout
        mask = torch.rand_like(x_aug) > self.feature_dropout
        x_aug = x_aug * mask

        # Small Gaussian noise
        noise = torch.randn_like(x_aug) * self.noise_std
        x_aug = x_aug + noise

        return x_aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        # Two augmented views of the same transaction
        x1 = self.augment(x)
        x2 = self.augment(x)

        return x1, x2


class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### 3.12 Create PyTorch DataLoaders

The SimCLR loaders are used for contrastive pretraining.

```python
BATCH_SIZE = 512

simclr_train_dataset = SimCLRTabularDataset(
    X_train_tensor,
    feature_dropout=0.15,
    noise_std=0.05
)

simclr_val_dataset = SimCLRTabularDataset(
    X_val_tensor,
    feature_dropout=0.15,
    noise_std=0.05
)

simclr_train_loader = DataLoader(
    simclr_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, # colab may run into issues with num_workers>0
    drop_last=True,
    pin_memory=True
)

simclr_val_loader = DataLoader(
    simclr_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0, # colab may run into issues with num_workers>0
    drop_last=True
)
```

The supervised loaders are used after SimCLR pretraining.

```python
train_dataset = FraudDataset(X_train_tensor, y_train_tensor)
val_dataset = FraudDataset(X_val_tensor, y_val_tensor)
test_dataset = FraudDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2
)
```

```python
len(train_loader), len(val_loader), len(test_loader)
```

### NOTE: We use following small sample for now

```python
from sklearn.model_selection import train_test_split
import torch

def stratified_tensor_subset(X, y, fraction=0.25, random_state=42):
    """
    Select a stratified subset of tensors.
    Keeps approximately the same fraud/non-fraud ratio.
    """
    indices = torch.arange(len(y)).numpy()

    subset_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=random_state,
        stratify=y.numpy()
    )

    subset_indices = torch.tensor(subset_indices, dtype=torch.long)

    return X[subset_indices], y[subset_indices]



X_train_25, y_train_25 = stratified_tensor_subset(
    X_train_tensor,
    y_train_tensor,
    fraction=0.10
)

X_val_25, y_val_25 = stratified_tensor_subset(
    X_val_tensor,
    y_val_tensor,
    fraction=0.10
)

X_test_25, y_test_25 = stratified_tensor_subset(
    X_test_tensor,
    y_test_tensor,
    fraction=0.10
)

print("Train 10%:", X_train_25.shape, y_train_25.shape, "fraud ratio:", y_train_25.mean().item())
print("Val 10%:  ", X_val_25.shape, y_val_25.shape, "fraud ratio:", y_val_25.mean().item())
print("Test 10%: ", X_test_25.shape, y_test_25.shape, "fraud ratio:", y_test_25.mean().item())
```

```python
simclr_train_dataset = SimCLRTabularDataset(
    X_train_25,
    feature_dropout=0.15,
    noise_std=0.05
)

simclr_val_dataset = SimCLRTabularDataset(
    X_val_25,
    feature_dropout=0.15,
    noise_std=0.05
)

simclr_test_dataset = SimCLRTabularDataset(
    X_test_25,
    feature_dropout=0.15,
    noise_std=0.05
)

simclr_train_loader = DataLoader(
    simclr_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, # colab may run into issues with num_workers>0
    drop_last=True,
    pin_memory=True
)

simclr_val_loader = DataLoader(
    simclr_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0, # colab may run into issues with num_workers>0
    drop_last=True
)


simclr_test_loader = DataLoader(
    simclr_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0, # colab may run into issues with num_workers>0
    drop_last=True
)
```

```python
train_dataset = FraudDataset(X_train_25, y_train_25)
val_dataset = FraudDataset(X_val_25, y_val_25)
test_dataset = FraudDataset(X_test_25, y_test_25)
```

The supervised loaders are used after SimCLR pretraining.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2
)
```

```python
len(train_loader), len(val_loader), len(test_loader)
```

### 3.13 Sanity-check DataLoader batches

Always check one batch before training.

```python
x1, x2 = next(iter(simclr_train_loader))

print("SimCLR batch:")
print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)

x, y = next(iter(train_loader))

print("\nSupervised batch:")
print("x shape:", x.shape)
print("y shape:", y.shape)

print("\nInput dimension:", x.shape[1])
print("Fraud ratio in supervised batch:", y.mean().item())
```

Expected output:

```text
SimCLR batch:
x1 shape: torch.Size([512, input_dim])
x2 shape: torch.Size([512, input_dim])

Supervised batch:
x shape: torch.Size([512, input_dim])
y shape: torch.Size([512])

Input dimension: input_dim
Fraud ratio in supervised batch: small number
```

### 3.14 Conceptual flow

```text
raw dataframe
   ↓
cleaned features and target
   ↓
train/val/test split
   ↓
scaling + encoding
   ↓
PyTorch tensors
   ↓
Dataset classes
   ↓
DataLoaders
   ↓
training loops
```

For SimCLR:

```text
one transaction x
   ↓
Dataset creates x1 and x2
   ↓
model learns x1 and x2 should be close
```

For supervised fraud detection:

```text
one transaction x
   ↓
model predicts isFraud
```

## Step 4. SimCLR method

In this step, we define the **SimCLR self-supervised learning method** using a PyTorch-style model design and then wrap the training logic inside a PyTorch Lightning model called `LitSimCLR`. The goal is to pretrain an encoder on unlabeled PaySim transactions. Each transaction is augmented twice by the `SimCLRTabularDataset`, producing `x1` and `x2`. These two views are treated as a positive pair, while other transactions in the same batch act as negatives. PyTorch models are typically built by subclassing `torch.nn.Module`, and Lightning organizes PyTorch training code into methods such as `training_step`, `validation_step`, and `configure_optimizers`. ([PyTorch Docs][1])

---

### 4.1 Install/import Lightning

```python
!pip install lightning -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
```

---

### 4.2 Define the tabular encoder

The encoder takes a preprocessed transaction vector and maps it into a lower-dimensional representation. This encoder is the part we want to reuse later for fraud classification.

```python
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)
```

Conceptually:

```text
transaction vector
   ↓
MLP encoder
   ↓
transaction embedding
```

---

### 4.3 Define the SimCLR projection head

SimCLR usually applies contrastive loss to the output of a **projection head**, not directly to the encoder embedding. The encoder representation is kept for downstream tasks, while the projection output is used for contrastive training.

```python
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, h):
        return self.net(h)
```

Conceptually:

```text
encoder embedding h
   ↓
projection head
   ↓
contrastive vector z
```

---

### 4.4 Define NT-Xent / InfoNCE loss

This loss pulls together two augmented views of the same transaction and pushes apart different transactions in the batch.

```python
import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.2):
    """
    NT-Xent loss used in contrastive learning / SimCLR.

    z1: projection vectors from augmented view 1
        shape: [batch_size, projection_dim]

    z2: projection vectors from augmented view 2
        shape: [batch_size, projection_dim]

    Goal:
        For each sample, make its two augmented views similar.
        At the same time, push it away from all other samples in the batch.
    """

    # Number of original samples in the batch
    batch_size = z1.size(0)

    # Normalize each vector to length 1.
    # After this, dot product between two vectors becomes cosine similarity.
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Stack both augmented views together.
    #
    # If batch_size = B:
    # z1 has shape [B, D]
    # z2 has shape [B, D]
    #
    # z will have shape [2B, D]
    #
    # Order:
    # index 0       -> z1[0]
    # index 1       -> z1[1]
    # ...
    # index B - 1   -> z1[B - 1]
    # index B       -> z2[0]
    # index B + 1   -> z2[1]
    # ...
    # index 2B - 1  -> z2[B - 1]
    z = torch.cat([z1, z2], dim=0)

    # Compute pairwise similarity between every vector and every other vector.
    #
    # z shape:   [2B, D]
    # z.T shape: [D, 2B]
    #
    # sim shape: [2B, 2B]
    #
    # sim[i][j] means:
    # similarity between z[i] and z[j]
    #
    # Dividing by temperature controls sharpness:
    # lower temperature = stronger penalty for wrong matches
    sim = torch.matmul(z, z.T) / temperature

    # Remove self-similarity.
    #
    # The diagonal contains similarity of each vector with itself:
    # sim[0][0], sim[1][1], sim[2][2], ...
    #
    # These are not useful because every vector is always most similar to itself.
    # So we replace the diagonal with a huge negative value.
    # During softmax, this becomes almost zero probability.
    self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    # Build the correct positive-pair index for every row.
    #
    # For the first half:
    # z1[i] should match z2[i]
    #
    # For the second half:
    # z2[i] should match z1[i]
    #
    # Example when batch_size = 4:
    #
    # z indices:
    # 0 -> z1[0]
    # 1 -> z1[1]
    # 2 -> z1[2]
    # 3 -> z1[3]
    # 4 -> z2[0]
    # 5 -> z2[1]
    # 6 -> z2[2]
    # 7 -> z2[3]
    #
    # Positive pairs:
    # 0 <-> 4
    # 1 <-> 5
    # 2 <-> 6
    # 3 <-> 7
    #
    # Therefore target labels should be:
    # [4, 5, 6, 7, 0, 1, 2, 3]
    positive_indices = torch.arange(2 * batch_size, device=z.device)
    positive_indices = (positive_indices + batch_size) % (2 * batch_size)

    # Cross entropy treats each row of sim as a classification problem.
    #
    # For each anchor vector z[i], the model must choose the correct positive
    # pair from all other vectors in the batch.
    #
    # Example:
    # row 0 should classify column 4 as the correct match
    # row 1 should classify column 5 as the correct match
    # row 4 should classify column 0 as the correct match
    #
    # This increases similarity with the positive pair
    # and decreases similarity with all negative pairs.
    loss = F.cross_entropy(sim, positive_indices)

    return loss
```

NT-Xent loss for anchor (z_i) and positive pair (z_j):

$\ell_{i,j}=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{[k\ne i]}\exp(\operatorname{sim}(z_i,z_k)/\tau)}$

Meaning:

```text
numerator   = similarity with the correct positive pair
denominator = similarity with all other vectors except itself
tau         = temperature
```

In the code:

```python
sim = torch.matmul(z, z.T) / temperature
```

computes all (\operatorname{sim}(z_i,z_k)/\tau).

```python
F.cross_entropy(sim, positive_indices)
```

applies:

```text
softmax + log + negative mean
```

So the goal is simple:

```text
make positive pairs close, negatives far
```

This works because each row of `sim` is treated like a classification problem: among all other samples in the batch, the correct answer is the matching augmented view.

---

### 4.5 Define useful SimCLR metrics

These are not fraud metrics yet. They help us monitor whether contrastive learning is behaving sensibly.

```python
import torch
import torch.nn.functional as F


def simclr_similarity_metrics(z1, z2):
    """
    Compute simple SimCLR similarity metrics.

    z1: projection vectors from augmented view 1
        shape: [batch_size, projection_dim]

    z2: projection vectors from augmented view 2
        shape: [batch_size, projection_dim]

    Returns:
        positive_similarity = average similarity between matching pairs
        negative_similarity = average similarity between non-matching pairs
        similarity_gap      = positive_similarity - negative_similarity

    Good behavior:
        positive_similarity should be high
        negative_similarity should be low
        similarity_gap should be large
    """

    # Normalize vectors so dot product becomes cosine similarity.
    #
    # sim(z_i, z_j) = z_i dot z_j
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    # Positive similarity:
    #
    # Compare matching augmented views:
    #
    # z1[0] with z2[0]
    # z1[1] with z2[1]
    # z1[2] with z2[2]
    # ...
    #
    # Equation:
    #
    # positive_sim = mean_i sim(z1[i], z2[i])
    positive_sim = torch.sum(z1 * z2, dim=1).mean()

    # Combine both views:
    #
    # z = [z1[0], z1[1], ..., z1[B-1],
    #      z2[0], z2[1], ..., z2[B-1]]
    #
    # Shape: [2B, D]
    z = torch.cat([z1, z2], dim=0)

    # Compute all pairwise cosine similarities.
    #
    # sim[i][j] = similarity between z[i] and z[j]
    #
    # Shape: [2B, 2B]
    sim = torch.matmul(z, z.T)

    # Mask for self-comparisons:
    #
    # sim[0][0], sim[1][1], sim[2][2], ...
    #
    # These compare a vector with itself, so we exclude them.
    self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)

    # Mask for positive pairs.
    #
    # If batch_size = 4:
    #
    # Positive pairs:
    # 0 <-> 4
    # 1 <-> 5
    # 2 <-> 6
    # 3 <-> 7
    #
    # These are matching augmented views, so we also exclude them
    # when computing negative similarity.
    positive_mask = torch.zeros_like(self_mask)

    for i in range(batch_size):
        positive_mask[i, i + batch_size] = True
        positive_mask[i + batch_size, i] = True

    # Negative pairs are everything except:
    #
    # 1. self-comparisons
    # 2. positive pairs
    #
    # So these represent non-matching examples.
    negative_mask = ~(self_mask | positive_mask)

    # Average similarity between all negative pairs.
    #
    # Good SimCLR training should make this low.
    negative_sim = sim[negative_mask].mean()

    # Gap between positive and negative similarity.
    #
    # Larger gap means the model separates positives from negatives better.
    similarity_gap = positive_sim - negative_sim

    return {
        "positive_similarity": positive_sim,
        "negative_similarity": negative_sim,
        "similarity_gap": similarity_gap
    }
```

During good pretraining, we usually want:

```text
positive_similarity ↑
negative_similarity lower
similarity_gap ↑
contrastive_loss ↓
```

---

### 4.6 Create the Lightning model: `LitSimCLR`

This is the main pretraining model. The encoder and projection head are normal PyTorch modules, while Lightning handles the training loop structure. Lightning’s `configure_optimizers()` is the standard place to define optimizers and schedulers. ([Lightning AI][2])

```python
class LitSimCLR(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        embedding_dim=128,
        projection_dim=64,
        dropout=0.2,
        temperature=0.2,
        lr=1e-3,
        weight_decay=1e-4
    ):
        super().__init__()

        # Saves hyperparameters automatically.
        # W&B/Lightning can use this to track model configuration for reproducibility.
        self.save_hyperparameters()

        # Encoder learns the reusable transaction representation.
        # This is the part we will later use for fraud classification.
        self.encoder = TabularEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # Projection head is used only for SimCLR contrastive training.
        # The loss is applied to projection z, not directly to encoder embedding h.
        self.projection_head = ProjectionHead(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim
        )

    def forward(self, x):
        """
        Return encoder embedding h.

        This embedding is the useful representation that will later be passed
        to a fraud classifier.
        """
        h = self.encoder(x)
        return h

    def project(self, x):
        """
        Return projection z used for contrastive learning.

        SimCLR applies NT-Xent loss on z, while h is kept as the reusable
        downstream representation.
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

    def shared_step(self, batch, stage):
        x1, x2 = batch

        # Two augmented views of the same transaction are passed through
        # the same encoder and projection head.
        h1, z1 = self.project(x1)
        h2, z2 = self.project(x2)

        # Main contrastive learning objective.
        # We log this because it tells us whether SimCLR pretraining is improving.
        loss = nt_xent_loss(
            z1,
            z2,
            temperature=self.hparams.temperature
        )

        # Similarity metrics help us understand whether the model is learning
        # to pull positive pairs together and push negative pairs apart.
        metrics = simclr_similarity_metrics(z1, z2)

        positive_similarity = metrics["positive_similarity"]
        negative_similarity = metrics["negative_similarity"]
        similarity_gap = metrics["similarity_gap"]

        # Embedding norm shows whether the encoder representation is exploding,
        # collapsing, or staying numerically stable.
        h1_norm = h1.norm(dim=1).mean()
        h2_norm = h2.norm(dim=1).mean()
        avg_embedding_norm = (h1_norm + h2_norm) / 2

        # Projection norm shows whether the contrastive space is numerically stable.
        # Since NT-Xent normalizes z internally, this is mostly a diagnostic metric.
        z1_norm = z1.norm(dim=1).mean()
        z2_norm = z2.norm(dim=1).mean()
        avg_projection_norm = (z1_norm + z2_norm) / 2

        # Standard deviation of embeddings helps detect representation collapse.
        # If this becomes extremely small, many samples may be mapped to similar vectors.
        embedding_std = torch.cat([h1, h2], dim=0).std(dim=0).mean()

        # Standard deviation of projections helps detect collapse in contrastive space.
        projection_std = torch.cat([z1, z2], dim=0).std(dim=0).mean()

        # Log contrastive loss.
        # Lower is generally better, but it should be interpreted with similarity metrics.
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        # Log positive similarity.
        # This should increase if two views of the same transaction are being pulled together.
        self.log(
            f"{stage}_positive_similarity",
            positive_similarity,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        # Log negative similarity.
        # This should remain lower than positive similarity.
        self.log(
            f"{stage}_negative_similarity",
            negative_similarity,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        # Log similarity gap.
        # This is one of the most useful SimCLR diagnostics:
        # positive_similarity - negative_similarity.
        # A larger gap usually means better contrastive separation.
        self.log(
            f"{stage}_similarity_gap",
            similarity_gap,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        # Log encoder embedding norm.
        # Helps check if encoder outputs are exploding or shrinking too much.
        self.log(
            f"{stage}_embedding_norm",
            avg_embedding_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=True
        )

        # Log projection norm.
        # Helps monitor the contrastive representation space.
        self.log(
            f"{stage}_projection_norm",
            avg_projection_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=True
        )

        # Log embedding standard deviation.
        # Very low values may indicate representation collapse.
        self.log(
            f"{stage}_embedding_std",
            embedding_std,
            prog_bar=False,
            on_step=True,
            on_epoch=True
        )

        # Log projection standard deviation.
        # Very low values may indicate projection collapse.
        self.log(
            f"{stage}_projection_std",
            projection_std,
            prog_bar=False,
            on_step=True,
            on_epoch=True
        )
        if stage == "train":
            # Log current learning rate.
            # Useful for debugging training behavior and comparing runs in W&B.
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

            self.log(
                f"{stage}_lr",
                current_lr,
                prog_bar=False,
                on_step=True,
                on_epoch=True
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer
```

---

### 4.7 Instantiate `LitSimCLR`

Use the input dimension from preprocessing:

```python
input_dim = X_train_tensor.shape[1]

lit_simclr = LitSimCLR(
    input_dim=input_dim,
    hidden_dim=256,
    embedding_dim=128,
    projection_dim=64,
    dropout=0.2,
    temperature=0.2,
    lr=1e-3,
    weight_decay=1e-4
)

print(lit_simclr)
```

```python
!pip install -q torchinfo

from torchinfo import summary

input_dim = X_train_tensor.shape[1]

summary(
    lit_simclr,
    input_size=(32, input_dim),   # example batch size = 32
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable"
    ],
    depth=4
)
```

---

### 4.8 Train with PyTorch Lightning

This uses the `simclr_train_loader` and `simclr_val_loader` from preprocessing.

```python
!pip install wandb lightning -q

import os
import wandb
from google.colab import userdata
from lightning.pytorch.loggers import WandbLogger

wandb_api_key = userdata.get("WANDB_API_KEY")

print("W&B key loaded:", wandb_api_key is not None)

wandb.login(key=wandb_api_key)
```

```python
PROJECT_NAME = "Fraud Detection with SimCLR Mini"
RUN_NAME = "Self-supervised pretraining stage - 01 Mini"

wandb_logger = WandbLogger(
    project=PROJECT_NAME,
    name=RUN_NAME,
    log_model=False
)
```

```python
len(simclr_train_loader), len(simclr_val_loader)
```

```python
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=1,
    logger=wandb_logger,
    log_every_n_steps=10,
    # limit_train_batches=100, # -- Comment this line
    # limit_val_batches=50, # -- Comment this line
)


trainer.fit(
    lit_simclr,
    train_dataloaders=simclr_train_loader,
    val_dataloaders=simclr_val_loader
)
```

| Metric                | Meaning                                                                | Desired behavior                             | What it may mean otherwise                                                                                            |
| --------------------- | ---------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `train_loss`          | NT-Xent / InfoNCE loss on training batches                             | Should decrease over time                    | If it stays flat, the encoder may not be learning. If it explodes, learning rate or loss computation may be unstable. |
| `val_loss`            | NT-Xent / InfoNCE loss on validation batches                           | Should decrease or stabilize                 | If train loss decreases but val loss increases, the model may be overfitting the training augmentations.              |
| `positive_similarity` | Average similarity between two augmented views of the same transaction | Should increase                              | If it stays low, the model is not pulling positive pairs together.                                                    |
| `negative_similarity` | Average similarity between views from different transactions           | Should stay lower than positive similarity   | If it becomes close to positive similarity, the model may not be separating different transactions well.              |
| `similarity_gap`      | `positive_similarity - negative_similarity`                            | Should increase                              | If it stays near zero, the model is not clearly distinguishing positives from negatives.                              |
| `embedding_norm`      | Average magnitude of encoder output `h`                                | Should remain stable                         | If it becomes very large, activations may be exploding. If it becomes near zero, representations may be collapsing.   |
| `projection_norm`     | Average magnitude of projection head output `z`                        | Should remain stable                         | If it becomes very large or near zero, the contrastive space may be unstable.                                         |
| `embedding_std`       | Average standard deviation across encoder embedding dimensions         | Should stay clearly above zero               | If it goes near zero, the encoder may be mapping many transactions to almost the same vector.                         |
| `projection_std`      | Average standard deviation across projection dimensions                | Should stay clearly above zero               | If it goes near zero, the projection head may be collapsing.                                                          |
| `train_lr`            | Current learning rate during training                                  | Should match the intended optimizer schedule | If loss is unstable, LR may be too high. If loss barely changes, LR may be too low.                                   |

For a healthy SimCLR run, the main pattern you want is:

```text
loss decreases
positive_similarity > negative_similarity
similarity_gap increases
embedding_std and projection_std do not collapse toward zero
```

---

### 4.9 Save the pretrained encoder

After training, save only the encoder because this is the reusable representation model.

```python
torch.save(lit_simclr.encoder.state_dict(), "simclr_encoder.pt")
```

```python
SIMCLR_ENCODER_MODEL_PATH = "/content/drive/MyDrive/DATA/CS5369L/SimCLR-Summer2026/SimCLR Encoder Fraud Detection.pt"
```

```python
try:
    torch.save(lit_simclr.encoder.state_dict(), SIMCLR_ENCODER_MODEL_PATH)
    print(f"Succesfully saved @ {SIMCLR_ENCODER_MODEL_PATH}")
except Exception as e:
    SIMCLR_ENCODER_MODEL_PATH = "simclr_encoder.pt"
    print(f"Failed to save @ {SIMCLR_ENCODER_MODEL_PATH}")
```

**Cleanup: Flush old GPU model safely**

```python
import torch

# Finish old W&B run if it is still active
try:
    wandb.finish()
except exception as e:
    print(e)
```

```python
import gc

# Delete old trainer/model objects if they exist
for var_name in [
    "trainer",
    "lit_simclr",
    "simclr_model",
    "model",
    "optimizer",
    "scheduler"]:
    if var_name in globals():
        del globals()[var_name]

# Run Python garbage collection
gc.collect()

# Clear PyTorch CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

print("GPU cache cleared.")
```

```python
## Check GPU Memory, just in case
if torch.cuda.is_available():
    print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
    print("Reserved: ", torch.cuda.memory_reserved() / 1024**3, "GB")
```

Later, the fraud classifier will use:

```text
transaction
   ↓
pretrained encoder
   ↓
classifier head
   ↓
fraud probability
```

---

## Step 5: Application - Training for Fraud Detection

**Training setup for supervised fraud classification**.

This assumes you already completed:

```text
Step 3: preprocessing + train_loader, val_loader, test_loader
Step 4: SimCLR pretraining + saved simclr_encoder.pt
```

---

### 5.0 Load the pretrained SimCLR encoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# Recreate the same encoder architecture used during SimCLR pretraining.
# This MUST match the architecture used when saving simclr_encoder.pt.
# If input_dim, hidden_dim, or embedding_dim are different, load_state_dict will fail.

pretrained_encoder = TabularEncoder(
    input_dim=X_train_tensor.shape[1],   # or X_train_25.shape[1] if using the 10% subset
    hidden_dim=256,
    embedding_dim=128,
    dropout=0.2
)


# Load the pretrained SimCLR encoder weights.
# map_location="cpu" makes loading safer even if the model was saved on GPU.
try:
    pretrained_encoder.load_state_dict(
    torch.load("simclr_encoder.pt", map_location="cpu"))
except:
    pretrained_encoder.load_state_dict(
    torch.load(SIMCLR_ENCODER_MODEL_PATH, map_location="cpu"))

# Put the encoder in training mode because we will fine-tune it.
pretrained_encoder.train()

print(f"Loaded pretrained SimCLR encoder from {SIMCLR_ENCODER_MODEL_PATH}")
```

---

### 5.1 Define the fraud classifier model

```python
class FraudClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim=128, hidden_dim=64, dropout=0.2):
        super().__init__()

        # The encoder comes from SimCLR pretraining.
        # It converts raw transaction features into a learned embedding.
        self.encoder = encoder

        # The classifier head maps the learned embedding to one output logit.
        # One logit is enough because this is binary classification:
        # fraud vs non-fraud.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Step 1: get encoder representation
        h = self.encoder(x)

        # Step 2: get fraud logit
        # Shape before squeeze: [batch_size, 1]
        logit = self.classifier(h).squeeze(1)

        # Return raw logit, not sigmoid probability.
        # BCEWithLogitsLoss expects raw logits.
        return logit
```

---

### 5.2 Create the fraud classifier using the pretrained encoder

```python
fraud_model = FraudClassifier(
    encoder=pretrained_encoder,
    embedding_dim=128,
    hidden_dim=64,
    dropout=0.2
)

print(fraud_model)
```

```python
input_dim = X_train_tensor.shape[1]

summary(
    fraud_model,
    input_size=(32, input_dim),   # example batch size = 32
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable"
    ],
    depth=4
)
```

---

### 5.3 Choose fine-tuning or frozen encoder

For the first experiment, I recommend **fine-tuning**.

```python
# Fine-tuning mode:
# The pretrained encoder continues learning during supervised fraud training.
for param in fraud_model.encoder.parameters():
    param.requires_grad = True

print("Encoder is trainable:", next(fraud_model.encoder.parameters()).requires_grad)
```

Optional frozen version for later ablation:

```python
# Optional ablation:
# Freeze encoder and train only the classifier head.
# for param in fraud_model.encoder.parameters():
#     param.requires_grad = False
```

---

### 5.4 Compute positive class weight

Fraud is rare, so we use `pos_weight` to make the loss care more about fraud examples.

```python
# Count positive and negative examples in the training labels.
num_positive = y_train_tensor.sum()
num_negative = len(y_train_tensor) - num_positive

# pos_weight tells BCEWithLogitsLoss how much more to weight the positive class.
# If fraud is rare, this value will be large.
pos_weight = num_negative / num_positive

print("Number of fraud examples:", int(num_positive.item()))
print("Number of non-fraud examples:", int(num_negative.item()))
print("Positive class weight:", pos_weight.item())
```

If you are using the 25% subset, use:

```python
# Use this instead if training on the 25% subset.
num_positive = y_train_25.sum()
num_negative = len(y_train_25) - num_positive
pos_weight = num_negative / num_positive

print("Number of fraud examples:", int(num_positive.item()))
print("Number of non-fraud examples:", int(num_negative.item()))
print("Positive class weight:", pos_weight.item())
```

---

### 5.5 Install and import metrics

```python

!pip install torchmetrics -q

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score
)
```

---

### 5.6 Define the Lightning model for supervised fraud classification

```python
class LitFraudClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        pos_weight,
        lr=1e-3,
        weight_decay=1e-4
    ):
        super().__init__()

        # Save hyperparameters for reproducibility.
        # We ignore "model" because it is a PyTorch object and does not serialize cleanly as a hyperparameter.
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        # BCEWithLogitsLoss combines sigmoid + binary cross entropy in one stable operation.
        # pos_weight helps with class imbalance by weighting fraud examples more.
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        # Validation metrics
        # ROC-AUC measures ranking quality across thresholds.
        self.val_auroc = BinaryAUROC()

        # PR-AUC is especially important for imbalanced fraud detection.
        self.val_auprc = BinaryAveragePrecision()

        # Threshold-based metrics use default threshold 0.5.
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        # Test metrics
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def forward(self, x):
        # Forward pass returns raw logits.
        return self.model(x)

    def shared_step(self, batch, stage):
        x, y = batch

        # Labels should be float for BCEWithLogitsLoss.
        y = y.float()

        # Raw model outputs.
        logits = self(x)

        # Supervised binary classification loss.
        loss = self.loss_fn(logits, y)

        # Convert logits to probabilities for metrics.
        probs = torch.sigmoid(logits)

        batch_size = x.size(0)

        # Log loss for train/val/test.
        # For training, on_step=True helps W&B update during the epoch.
        if stage == "train":
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size
            )

            # Log learning rate during training for debugging optimization.
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(
                "train_lr",
                current_lr,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size
            )

        else:
            self.log(
                f"{stage}_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size
            )

        # Update validation metrics.
        # These are computed over the entire validation epoch.
        if stage == "val":
            self.val_auroc.update(probs, y.int())
            self.val_auprc.update(probs, y.int())
            self.val_precision.update(probs, y.int())
            self.val_recall.update(probs, y.int())
            self.val_f1.update(probs, y.int())

        # Update test metrics.
        elif stage == "test":
            self.test_auroc.update(probs, y.int())
            self.test_auprc.update(probs, y.int())
            self.test_precision.update(probs, y.int())
            self.test_recall.update(probs, y.int())
            self.test_f1.update(probs, y.int())

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="test")

    def on_validation_epoch_end(self):
        # Compute metrics accumulated over all validation batches.
        val_roc_auc = self.val_auroc.compute()
        val_pr_auc = self.val_auprc.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()

        # Log validation metrics.
        # PR-AUC is very important for imbalanced fraud detection.
        self.log("val_roc_auc", val_roc_auc, prog_bar=True)
        self.log("val_pr_auc", val_pr_auc, prog_bar=True)
        self.log("val_precision", val_precision)
        self.log("val_recall", val_recall)
        self.log("val_f1", val_f1, prog_bar=True)

        # Reset metrics before the next epoch.
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        # Compute metrics accumulated over all test batches.
        test_roc_auc = self.test_auroc.compute()
        test_pr_auc = self.test_auprc.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()

        # Log final test metrics.
        self.log("test_roc_auc", test_roc_auc)
        self.log("test_pr_auc", test_pr_auc)
        self.log("test_precision", test_precision)
        self.log("test_recall", test_recall)
        self.log("test_f1", test_f1)

        # Reset metrics.
        self.test_auroc.reset()
        self.test_auprc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        # AdamW is a strong default optimizer for neural networks.
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer
```

---

### 5.7 Create the Lightning classifier

```python

lit_fraud = LitFraudClassifier(
    model=fraud_model,
    pos_weight=pos_weight,
    lr=1e-3,
    weight_decay=1e-4
)

lit_fraud.train()

print(lit_fraud)
```

```python
input_dim = X_train_tensor.shape[1]

summary(
    lit_fraud,
    input_size=(32, input_dim),   # example batch size = 32
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable"
    ],
    depth=4
)
```

---

### 5.8 Login to W&B and create logger

Use this if your W&B key is saved in Colab Secrets as `WANDB_API_KEY`.

```python
!pip install wandb -q

import os
import wandb
from google.colab import userdata
from lightning.pytorch.loggers import WandbLogger

# Load W&B API key from Colab Secrets.
wandb_api_key = userdata.get("WANDB_API_KEY")

print("W&B key loaded:", wandb_api_key is not None)

# Login to W&B.
wandb.login(key=wandb_api_key)

PROJECT_NAME = "Fraud Classification with SimCLR Encoder - mini"
RUN_NAME = "Supervised fraud classification stage - v1"

wandb_logger_cls = WandbLogger(
    project=PROJECT_NAME,
    name=RUN_NAME,
    log_model=False
)
```

---

### 5.9 Train the fraud classifier

```python

trainer_cls = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    logger=wandb_logger_cls,
    log_every_n_steps=10,
    enable_progress_bar=True
)

trainer_cls.fit(
    lit_fraud,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)
```

For a quick debug run, use:

```python
# Debug trainer for faster testing.
# trainer_cls = pl.Trainer(
#     max_epochs=2,
#     accelerator="auto",
#     devices="auto",
#     logger=wandb_logger_cls,
#     log_every_n_steps=10,
#     limit_train_batches=500,
#     limit_val_batches=100,
#     precision="32-true",
#     enable_progress_bar=True
# )
```

---

### 5.10 Test the fraud classifier

```python
trainer_cls.test(
    lit_fraud,
    dataloaders=test_loader
)
```

---

### 5.11 Save the supervised fraud classifier

```python
# Save the full fraud classifier model.
torch.save(
    lit_fraud.model.state_dict(),
    "simclr_fraud_classifier.pt"
)

print("Saved simclr_fraud_classifier.pt")
```

---

### 5.12 Log the classifier as a W&B artifact

```python
artifact = wandb.Artifact(
    name="simclr-fraud-classifier",
    type="model",
    description="Fraud classifier using a SimCLR-pretrained tabular encoder"
)

artifact.add_file("simclr_fraud_classifier.pt")

wandb_logger_cls.experiment.log_artifact(artifact)

print("Logged classifier artifact to W&B")
```

---

### 5.13 Finish the W&B run

```python
wandb.finish()
```

---

### 5.14 What we are doing in this step

In this supervised training stage, we take the encoder learned during SimCLR pretraining and reuse it as the backbone for fraud detection. We attach a small classifier head that outputs one fraud logit per transaction. The model is trained using `BCEWithLogitsLoss`, with positive class weighting to handle the rarity of fraud examples. During validation and testing, we track ROC-AUC, PR-AUC, precision, recall, and F1-score. The main goal is to see whether the SimCLR-pretrained encoder gives us useful representations for the downstream fraud classification task.

These results are actually very informative. The model is **ranking fraud cases well**, but its **default classification threshold is producing too many false positives**.

### What the testing line means

```text
Testing 94/94 110.17it/s
```

This means your `test_loader` had **94 batches**, and Lightning finished evaluating all of them. Since you are using only **10% of the data**, this is a smaller test set than the full PaySim test set, so the results are useful for debugging but should not be treated as final full-data performance.

---

### Meaning of each metric

| Metric           |    Value | Meaning                                                                                                 |
| ---------------- | -------: | ------------------------------------------------------------------------------------------------------- |
| `test_loss`      | `0.5912` | Binary classification loss. Lower is better, but by itself it is not the main fraud metric.             |
| `test_roc_auc`   | `0.9416` | Very good ranking ability. The model usually assigns higher scores to fraud cases than non-fraud cases. |
| `test_pr_auc`    | `0.1948` | Moderate for imbalanced fraud detection. Better than random, but there is room to improve.              |
| `test_recall`    | `0.8699` | The model catches about **87% of fraud cases**.                                                         |
| `test_precision` | `0.0083` | Very low. Among transactions predicted as fraud, less than **1% are actually fraud**.                   |
| `test_f1`        | `0.0164` | Very low because precision is extremely low.                                                            |

---

### Main interpretation

Your model is currently behaving like this:

```text
It catches many fraud cases,
but it also flags a huge number of legitimate transactions as fraud.
```

That is why recall is high but precision is very low.

In fraud detection language:

```text
High recall = catches most fraud
Low precision = creates many false alarms
```

So the model is sensitive, but not selective.

---

### Why ROC-AUC is high but precision is low

This is common in imbalanced fraud detection.

`ROC-AUC = 0.94` means the model is good at **ranking** fraud above non-fraud. But your precision, recall, and F1 are computed using a default threshold, usually:

```text
threshold = 0.5
```

That threshold may be bad for this dataset.

So the model may have useful scores, but the cutoff for saying “fraud” needs tuning.

---

### Why PR-AUC matters more here

Your `test_pr_auc` is:

```text
0.1948
```

This is more realistic than ROC-AUC for fraud detection because fraud is rare. PR-AUC focuses on performance for the positive class, fraud.

A PR-AUC of `0.1948` means the model is doing better than random, but the current fraud predictions are still noisy.

---

### What to do next

The next step should be **threshold tuning** on the validation set.

Instead of using:

```text
threshold = 0.5
```

find a threshold that gives a better precision-recall tradeoff.

For example:

```text
choose threshold that maximizes F1
or
choose threshold where precision >= 0.80
or
choose threshold where recall is high but false alarms are acceptable
```

Right now, your results suggest:

```text
The encoder/classifier learned useful ranking information,
but the decision threshold is not calibrated.
```

### Bottom line

For a 10% experiment, this is a promising early result:

```text
ROC-AUC is strong: 0.94
Recall is high: 0.87
Precision is poor: 0.008
```

So the model is **finding fraud-like patterns**, but it needs **threshold tuning, calibration, and possibly stronger supervised training or baseline comparison** before the predictions are practically useful.

```python
model = lit_fraud.model
```

```python
model = lit_fraud.model
```

### Evaluate higher thresholds using TorchMetrics

```python
import torch
import pandas as pd

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = lit_fraud.model.to(device)
model.eval()

all_probs = []
all_targets = []

# Collect probabilities and labels from test set
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).int()

        # Model returns raw logits
        logits = model(x)

        # Convert logits to fraud probabilities
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())

all_probs = torch.cat(all_probs)
all_targets = torch.cat(all_targets)

print("Prob shape:", all_probs.shape)
print("Target shape:", all_targets.shape)
print("Fraud ratio:", all_targets.float().mean().item())
```

### Threshold-independent metrics

These do **not** depend on a fixed threshold:

```python
roc_auc_metric = BinaryAUROC()
pr_auc_metric = BinaryAveragePrecision()

roc_auc = roc_auc_metric(all_probs, all_targets)
pr_auc = pr_auc_metric(all_probs, all_targets)

print("ROC-AUC:", roc_auc.item())
print("PR-AUC:", pr_auc.item())
```

### Threshold-based metrics

These depend on the fraud decision cutoff:

```python
thresholds = [
    0.50, 0.60, 0.70, 0.80, 0.90,
    0.95, 0.97, 0.98, 0.99, 0.995
]

rows = []

for threshold in thresholds:
    precision_metric = BinaryPrecision(threshold=threshold)
    recall_metric = BinaryRecall(threshold=threshold)
    f1_metric = BinaryF1Score(threshold=threshold)
    cm_metric = BinaryConfusionMatrix(threshold=threshold)

    precision = precision_metric(all_probs, all_targets)
    recall = recall_metric(all_probs, all_targets)
    f1 = f1_metric(all_probs, all_targets)

    cm = cm_metric(all_probs, all_targets)

    # Confusion matrix format:
    # [[TN, FP],
    #  [FN, TP]]
    tn = cm[0, 0].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()
    tp = cm[1, 1].item()

    predicted_fraud_count = int((all_probs >= threshold).sum().item())

    rows.append({
        "threshold": threshold,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_fraud_count": predicted_fraud_count
    })

threshold_df = pd.DataFrame(rows)

threshold_df
```

```python
# ============================================================
# Threshold Evaluation + Modern Visualization for Fraud Model
# ============================================================

import torch
import pandas as pd
import plotly.graph_objects as go

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix
)

# ------------------------------------------------------------
# 1. Use the trained model already in memory
# ------------------------------------------------------------
# lit_fraud.model is the trained FraudClassifier from your Lightning model.
# This avoids reloading from disk or W&B artifact.

device = "cuda" if torch.cuda.is_available() else "cpu"

model = lit_fraud.model.to(device)
model.eval()

print("Using device:", device)


# ------------------------------------------------------------
# 2. Collect predicted probabilities and true labels
# ------------------------------------------------------------
# The model outputs raw logits.
# We apply sigmoid to convert logits into fraud probabilities.

all_probs = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).int()

        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())

all_probs = torch.cat(all_probs)
all_targets = torch.cat(all_targets)

print("Probability shape:", all_probs.shape)
print("Target shape:", all_targets.shape)
print("Fraud ratio in test set:", all_targets.float().mean().item())


# ------------------------------------------------------------
# 3. Compute threshold-independent metrics
# ------------------------------------------------------------
# ROC-AUC and PR-AUC do not use one fixed threshold.
# They evaluate how well the model ranks fraud above non-fraud.

roc_auc_metric = BinaryAUROC()
pr_auc_metric = BinaryAveragePrecision()

roc_auc = roc_auc_metric(all_probs, all_targets)
pr_auc = pr_auc_metric(all_probs, all_targets)

print("ROC-AUC:", roc_auc.item())
print("PR-AUC:", pr_auc.item())


# ------------------------------------------------------------
# 4. Compute threshold-based metrics
# ------------------------------------------------------------
# Precision, recall, F1, and confusion matrix depend on a threshold.
# Higher threshold usually means fewer fraud alerts, higher precision,
# lower recall, and fewer false positives.

thresholds = [
    0.50, 0.60, 0.70, 0.80, 0.90,
    0.95, 0.97, 0.98, 0.99, 0.995
]

rows = []

for threshold in thresholds:
    precision_metric = BinaryPrecision(threshold=threshold)
    recall_metric = BinaryRecall(threshold=threshold)
    f1_metric = BinaryF1Score(threshold=threshold)
    cm_metric = BinaryConfusionMatrix(threshold=threshold)

    precision = precision_metric(all_probs, all_targets)
    recall = recall_metric(all_probs, all_targets)
    f1 = f1_metric(all_probs, all_targets)

    cm = cm_metric(all_probs, all_targets)

    # TorchMetrics binary confusion matrix format:
    # [[TN, FP],
    #  [FN, TP]]
    tn = int(cm[0, 0].item())
    fp = int(cm[0, 1].item())
    fn = int(cm[1, 0].item())
    tp = int(cm[1, 1].item())

    predicted_fraud_count = int((all_probs >= threshold).sum().item())

    rows.append({
        "threshold": threshold,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_fraud_count": predicted_fraud_count
    })

threshold_df = pd.DataFrame(rows)

print("Threshold evaluation table:")
display(threshold_df)


# ------------------------------------------------------------
# 5. Find best threshold by F1-score
# ------------------------------------------------------------
# F1 balances precision and recall.
# This may not always be the best business threshold, but it is useful.

best_idx = threshold_df["f1"].idxmax()
best_row = threshold_df.loc[best_idx]

print("Best threshold by F1:")
display(best_row)


# ------------------------------------------------------------
# 6. Plot precision, recall, and F1 vs threshold
# ------------------------------------------------------------
# This is the most important threshold plot.
# It shows the precision-recall tradeoff.

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=threshold_df["threshold"],
    y=threshold_df["precision"],
    mode="lines+markers",
    name="Precision"
))

fig.add_trace(go.Scatter(
    x=threshold_df["threshold"],
    y=threshold_df["recall"],
    mode="lines+markers",
    name="Recall"
))

fig.add_trace(go.Scatter(
    x=threshold_df["threshold"],
    y=threshold_df["f1"],
    mode="lines+markers",
    name="F1-score"
))

fig.update_layout(
    title="Precision, Recall, and F1 Across Fraud Thresholds",
    xaxis_title="Decision threshold",
    yaxis_title="Metric value",
    template="plotly_white",
    width=950,
    height=550,
    hovermode="x unified"
)

fig.show()


# ------------------------------------------------------------
# 7. Plot false positives and false negatives vs threshold
# ------------------------------------------------------------
# False positives = legitimate transactions incorrectly flagged as fraud.
# False negatives = fraud transactions missed by the model.

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=threshold_df["threshold"],
    y=threshold_df["fp"],
    mode="lines+markers",
    name="False positives"
))

fig.add_trace(go.Scatter(
    x=threshold_df["threshold"],
    y=threshold_df["fn"],
    mode="lines+markers",
    name="False negatives"
))

fig.update_layout(
    title="False Positives and False Negatives Across Thresholds",
    xaxis_title="Decision threshold",
    yaxis_title="Count",
    template="plotly_white",
    width=950,
    height=550,
    hovermode="x unified"
)

fig.show()


# ------------------------------------------------------------
# 8. Plot predicted fraud count vs threshold
# ------------------------------------------------------------
# This shows how many transactions the model flags as fraud at each threshold.

fig = go.Figure()

fig.add_trace(go.Bar(
    x=threshold_df["threshold"].astype(str),
    y=threshold_df["predicted_fraud_count"],
    name="Predicted fraud count"
))

fig.update_layout(
    title="Number of Transactions Flagged as Fraud by Threshold",
    xaxis_title="Decision threshold",
    yaxis_title="Predicted fraud count",
    template="plotly_white",
    width=950,
    height=550
)

fig.show()


# ------------------------------------------------------------
# 9. Plot confusion matrix components across thresholds
# ------------------------------------------------------------
# TP = fraud correctly detected
# FP = legitimate transaction incorrectly flagged
# TN = legitimate transaction correctly ignored
# FN = fraud missed

fig = go.Figure()

for col in ["tp", "fp", "tn", "fn"]:
    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df[col],
        mode="lines+markers",
        name=col.upper()
    ))

fig.update_layout(
    title="Confusion Matrix Components Across Thresholds",
    xaxis_title="Decision threshold",
    yaxis_title="Count",
    template="plotly_white",
    width=950,
    height=550,
    hovermode="x unified"
)

fig.show()


# ------------------------------------------------------------
# 10. Highlight best F1 threshold
# ------------------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=threshold_df["threshold"],
    y=threshold_df["f1"],
    mode="lines+markers",
    name="F1-score"
))

fig.add_trace(go.Scatter(
    x=[best_row["threshold"]],
    y=[best_row["f1"]],
    mode="markers+text",
    name="Best F1",
    text=[f"Best F1<br>threshold={best_row['threshold']}"],
    textposition="top center",
    marker=dict(size=12)
))

fig.update_layout(
    title="Best F1 Threshold",
    xaxis_title="Decision threshold",
    yaxis_title="F1-score",
    template="plotly_white",
    width=950,
    height=550
)

fig.show()


# ------------------------------------------------------------
# 11. Optional: choose threshold by minimum precision requirement
# ------------------------------------------------------------
# In fraud detection, you may want a minimum precision to control false alarms.
# Example: find the threshold with highest recall while precision >= 0.10.

min_precision = 0.10

valid = threshold_df[threshold_df["precision"] >= min_precision]

if len(valid) > 0:
    best_recall_at_precision = valid.loc[valid["recall"].idxmax()]
    print(f"Best recall where precision >= {min_precision}:")
    display(best_recall_at_precision)
else:
    print(f"No threshold reached precision >= {min_precision}")


# ------------------------------------------------------------
# 12. Optional: log threshold table to W&B
# ------------------------------------------------------------
# Uncomment if you want to log this analysis to W&B.

# import wandb
#
# wandb.init(
#     project="Fraud Classification with SimCLR Encoder - mini",
#     name="Threshold visualization"
# )
#
# wandb.log({
#     "test_roc_auc": roc_auc.item(),
#     "test_pr_auc": pr_auc.item(),
#     "threshold_metrics_table": wandb.Table(dataframe=threshold_df)
# })
#
# wandb.finish()
```

```python
# ============================================================
# ROC Curve with Thresholds in One Place
# ============================================================

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score

# ------------------------------------------------------------
# 1. Convert tensors to numpy if needed
# ------------------------------------------------------------

y_true = all_targets.numpy() if hasattr(all_targets, "numpy") else all_targets
y_score = all_probs.numpy() if hasattr(all_probs, "numpy") else all_probs


# ------------------------------------------------------------
# 2. Compute ROC curve and ROC-AUC
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)

print("ROC-AUC:", roc_auc)


# ------------------------------------------------------------
# 3. Create hover text for every ROC point
# ------------------------------------------------------------
# Each threshold creates one point on the ROC curve:
# threshold -> FPR, TPR

hover_text = [
    f"Threshold: {thr:.4f}<br>FPR: {x:.4f}<br>TPR/Recall: {y:.4f}"
    for x, y, thr in zip(fpr, tpr, thresholds)
]


# ------------------------------------------------------------
# 4. Create ROC curve figure
# ------------------------------------------------------------

fig = go.Figure()

# Main ROC curve
fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode="lines",
    name=f"ROC Curve, AUC = {roc_auc:.4f}",
    text=hover_text,
    hoverinfo="text",
    line=dict(width=4)
))

# Random classifier diagonal line
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    name="Random classifier",
    line=dict(width=2, dash="dash")
))


# ------------------------------------------------------------
# 5. Add markers for selected thresholds
# ------------------------------------------------------------

selected_thresholds = [0.5, 0.7, 0.9, 0.95, 0.99, 0.995]

marker_x = []
marker_y = []
marker_labels = []
marker_hover_text = []

for target_threshold in selected_thresholds:
    # Find closest threshold available in sklearn's ROC threshold array
    idx = np.argmin(np.abs(thresholds - target_threshold))

    marker_x.append(fpr[idx])
    marker_y.append(tpr[idx])
    marker_labels.append(f"{target_threshold}")

    marker_hover_text.append(
        f"Requested threshold: {target_threshold:.3f}<br>"
        f"Closest ROC threshold: {thresholds[idx]:.4f}<br>"
        f"FPR: {fpr[idx]:.4f}<br>"
        f"TPR/Recall: {tpr[idx]:.4f}"
    )

fig.add_trace(go.Scatter(
    x=marker_x,
    y=marker_y,
    mode="markers+text",
    name="Selected thresholds",
    text=marker_labels,
    textposition="top center",
    hovertext=marker_hover_text,
    hoverinfo="text",
    marker=dict(size=10)
))


# ------------------------------------------------------------
# 6. Final layout
# ------------------------------------------------------------

fig.update_layout(
    title="ROC Curve with Thresholds",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate / Recall",
    template="plotly_white",
    width=900,
    height=650,
    hovermode="closest"
)

fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])

fig.show()


# ------------------------------------------------------------
# 7. Optional: print selected threshold table
# ------------------------------------------------------------

rows = []

for target_threshold in selected_thresholds:
    idx = np.argmin(np.abs(thresholds - target_threshold))

    rows.append({
        "requested_threshold": target_threshold,
        "closest_roc_threshold": thresholds[idx],
        "fpr": fpr[idx],
        "tpr_recall": tpr[idx]
    })

roc_threshold_df = pd.DataFrame(rows)

display(roc_threshold_df)
```

```python
# ============================================================
# Precision-Recall Curve with Thresholds in One Place
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

# ------------------------------------------------------------
# 1. Convert tensors to numpy if needed
# ------------------------------------------------------------

y_true = all_targets.numpy() if hasattr(all_targets, "numpy") else all_targets
y_score = all_probs.numpy() if hasattr(all_probs, "numpy") else all_probs


# ------------------------------------------------------------
# 2. Compute Precision-Recall curve and PR-AUC
# ------------------------------------------------------------
# precision_recall_curve returns:
# precision: length = len(thresholds) + 1
# recall:    length = len(thresholds) + 1
# thresholds: length = len(precision) - 1
#
# So thresholds[i] corresponds to precision[i] and recall[i].

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

print("PR-AUC / Average Precision:", pr_auc)


# ------------------------------------------------------------
# 3. Create hover text
# ------------------------------------------------------------
# Since precision and recall have one extra point, we use only
# precision[:-1] and recall[:-1] for threshold-based hover text.

hover_text = [
    f"Threshold: {thr:.4f}<br>Recall: {rec:.4f}<br>Precision: {prec:.4f}"
    for rec, prec, thr in zip(recall[:-1], precision[:-1], thresholds)
]


# ------------------------------------------------------------
# 4. Create PR curve figure
# ------------------------------------------------------------

fig = go.Figure()

# PR curve with threshold hover
fig.add_trace(go.Scatter(
    x=recall[:-1],
    y=precision[:-1],
    mode="lines",
    name=f"PR Curve, AP = {pr_auc:.4f}",
    text=hover_text,
    hoverinfo="text",
    line=dict(width=4)
))


# ------------------------------------------------------------
# 5. Add random baseline
# ------------------------------------------------------------
# For PR curve, random baseline is approximately the fraud rate.

fraud_rate = y_true.mean()

fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[fraud_rate, fraud_rate],
    mode="lines",
    name=f"Random baseline, fraud rate = {fraud_rate:.4f}",
    line=dict(width=2, dash="dash")
))


# ------------------------------------------------------------
# 6. Add markers for selected thresholds
# ------------------------------------------------------------

selected_thresholds = [0.5, 0.7, 0.9, 0.95, 0.99, 0.995]

marker_x = []
marker_y = []
marker_labels = []
marker_hover_text = []

for target_threshold in selected_thresholds:
    # Find closest threshold available in sklearn's PR threshold array
    idx = np.argmin(np.abs(thresholds - target_threshold))

    marker_x.append(recall[idx])
    marker_y.append(precision[idx])
    marker_labels.append(f"{target_threshold}")

    marker_hover_text.append(
        f"Requested threshold: {target_threshold:.3f}<br>"
        f"Closest PR threshold: {thresholds[idx]:.4f}<br>"
        f"Recall: {recall[idx]:.4f}<br>"
        f"Precision: {precision[idx]:.4f}"
    )

fig.add_trace(go.Scatter(
    x=marker_x,
    y=marker_y,
    mode="markers+text",
    name="Selected thresholds",
    text=marker_labels,
    textposition="top center",
    hovertext=marker_hover_text,
    hoverinfo="text",
    marker=dict(size=10)
))


# ------------------------------------------------------------
# 7. Final layout
# ------------------------------------------------------------

fig.update_layout(
    title="Precision-Recall Curve with Thresholds",
    xaxis_title="Recall",
    yaxis_title="Precision",
    template="plotly_white",
    width=900,
    height=650,
    hovermode="closest"
)

fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])

fig.show()


# ------------------------------------------------------------
# 8. Optional: print selected threshold table
# ------------------------------------------------------------

rows = []

for target_threshold in selected_thresholds:
    idx = np.argmin(np.abs(thresholds - target_threshold))

    rows.append({
        "requested_threshold": target_threshold,
        "closest_pr_threshold": thresholds[idx],
        "recall": recall[idx],
        "precision": precision[idx]
    })

pr_threshold_df = pd.DataFrame(rows)

display(pr_threshold_df)
```

### Find best threshold by F1

```python
best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_f1_row
```

### Find best recall with minimum precision

```python
min_precision = 0.10

valid = threshold_df[threshold_df["precision"] >= min_precision]

if len(valid) > 0:
    best_row = valid.loc[valid["recall"].idxmax()]
    print(f"Best recall where precision >= {min_precision}:")
    display(best_row)
else:
    print(f"No threshold reached precision >= {min_precision}")
```

```python
min_precision = 0.10

valid = threshold_df[threshold_df["precision"] >= min_precision]

if len(valid) > 0:
    best_row = valid.loc[valid["recall"].idxmax()]
    print(f"Best recall where precision >= {min_precision}:")
    display(best_row)
else:
    print(f"No threshold reached precision >= {min_precision}")
```

The key idea:

```text
ROC-AUC and PR-AUC evaluate ranking quality.
Precision, recall, F1, and confusion matrix evaluate a specific threshold.
```

So for your previous result, increasing the threshold should usually:

```text
increase precision
decrease recall
reduce false positives
reduce predicted_fraud_count
```

This table shows how your fraud classifier behaves when you change the **decision threshold**.

The model outputs a fraud probability between `0` and `1`. The threshold decides when to call something fraud:

```text
if probability >= threshold -> predict fraud
if probability < threshold  -> predict non-fraud
```

As you increase the threshold, the model becomes more conservative.

---

### Main pattern

Your table shows this expected tradeoff:

| As threshold increases          | What happens                                          |
| ------------------------------- | ----------------------------------------------------- |
| Precision increases             | Fraud alerts become more trustworthy                  |
| Recall decreases                | You miss more real fraud cases                        |
| False positives decrease        | Fewer legitimate transactions are incorrectly flagged |
| False negatives increase        | More fraud transactions are missed                    |
| Predicted fraud count decreases | Model flags fewer transactions as fraud               |

---

### Example at threshold `0.50`

```text
precision = 0.0083
recall = 0.8699
fp = 12789
tp = 107
fn = 16
predicted fraud = 12896
```

Meaning:

The model catches **107 out of 123 fraud cases**, so recall is high. But it also falsely flags **12,789 legitimate transactions** as fraud. Precision is extremely low because most fraud alerts are false alarms.

So threshold `0.50` is too aggressive.

---

### Example at threshold `0.995`

```text
precision = 0.5882
recall = 0.1626
fp = 14
tp = 20
fn = 103
predicted fraud = 34
```

Meaning:

The model flags only **34 transactions** as fraud. Out of those, **20 are actually fraud**, so precision is much better. But it only catches **20 out of 123 fraud cases**, so recall is low.

So threshold `0.995` is very conservative.

---

### Best F1 in this table

The highest F1 appears at:

```text
threshold = 0.995
f1 = 0.2548
```

This gives the best balance between precision and recall among the thresholds you tested.

But whether this is “best” depends on your goal.

---

### If your goal is to catch most fraud

Use a lower threshold, such as:

```text
0.80 or 0.90
```

At `0.90`:

```text
precision = 0.0422
recall = 0.4878
tp = 60
fp = 1363
fn = 63
```

You catch about half the fraud cases, but still create many false alarms.

---

### If your goal is fewer false alarms

Use a higher threshold, such as:

```text
0.99 or 0.995
```

At `0.99`:

```text
precision = 0.2857
recall = 0.2114
tp = 26
fp = 65
fn = 97
```

This is more selective. About 29% of fraud alerts are real fraud.

---

### Bottom line

Your model is learning useful fraud scores because raising the threshold improves precision and reduces false positives. But the model still has a strong precision-recall tradeoff:

```text
Low threshold  -> catches many frauds, too many false alarms
High threshold -> fewer false alarms, misses many frauds
```

For a practical fraud detector, I would report multiple operating points, for example:

```text
threshold 0.90  -> higher recall mode
threshold 0.99  -> balanced alert mode
threshold 0.995 -> high precision mode
```

### **Handy notes: ROC curve vs Precision-Recall curve**

### 1. ROC curve

**Full form:** Receiver Operating Characteristic curve.

“Receiver operating” comes from old signal-detection/radar language: a receiver had to decide whether a signal was real or just noise. In ML, the model is doing the same thing: deciding whether a sample is positive or negative.

ROC curve plots:

```text
x-axis = FPR = False Positive Rate
y-axis = TPR = True Positive Rate = Recall
```

A point on ROC curve means:

```text
At this threshold:
x = fraction of normal cases wrongly flagged
y = fraction of fraud cases correctly caught
```

Example:

```text
FPR = 0.05, TPR = 0.80
```

Meaning:

```text
Model catches 80% of fraud,
but wrongly flags 5% of legitimate transactions.
```

Ideal ROC shape:

```text
goes close to top-left corner
```

Ideal ROC-AUC:

```text
1.0 = perfect
0.5 = random
```

ROC-AUC is useful when you care about **overall ranking/separation** between positive and negative classes.

---

### 2. Precision-Recall curve

PR curve plots:

```text
x-axis = Recall
y-axis = Precision
```

A point on PR curve means:

```text
At this threshold:
x = how much fraud the model catches
y = how trustworthy the fraud alerts are
```

Example:

```text
Recall = 0.70, Precision = 0.40
```

Meaning:

```text
Model catches 70% of fraud,
and 40% of predicted fraud alerts are actually fraud.
```

Ideal PR shape:

```text
stays near the top-right corner
```

Ideal PR-AUC:

```text
1.0 = perfect
baseline = positive class rate
```

For fraud detection, PR curve is often more important because fraud is rare.

---

### 3. Threshold idea

Both curves are created by changing the threshold.

```text
high threshold -> stricter model
low threshold  -> looser model
```

High threshold:

```text
fewer fraud alerts
higher precision
lower recall
```

Low threshold:

```text
more fraud alerts
higher recall
lower precision
```

The threshold is not usually shown on the axis. Each point secretly corresponds to one threshold.

---

### 4. When to use which?

| Metric/Curve    | Best for                                | Why                                                       |
| --------------- | --------------------------------------- | --------------------------------------------------------- |
| ROC curve       | General ranking quality                 | Shows whether positives score higher than negatives       |
| PR curve        | Rare positive class, like fraud/disease | Focuses on whether positive predictions are useful        |
| Threshold table | Deployment decision                     | Shows exact precision, recall, FP, FN at chosen threshold |

### Simple takeaway

```text
ROC curve asks:
Can the model separate fraud from non-fraud?

PR curve asks:
When the model says fraud, how useful is that alert?

Threshold table asks:
Which operating point should we actually use?
```

### **Hidden threshold idea in ROC and PR curves**

The **threshold is not on the x-axis or y-axis**.

Instead:

```text
Each point on the curve is produced by one threshold.
```

The curve is created by sweeping the threshold from **very high** to **very low**.

---

## 1. General threshold movement

Prediction rule:

```text
if probability >= threshold -> predict fraud
```

### Very high threshold

```text
threshold = 0.9999
```

Model is very strict.

```text
Predicts almost nothing as fraud.
TP is low, FP is low.
```

### Very low threshold

```text
threshold = 0.001
```

Model is very loose.

```text
Predicts many things as fraud.
TP is high, FP is high.
```

---

## 2. ROC curve threshold direction

ROC axes:

```text
x-axis = FPR
y-axis = TPR / Recall
```

Threshold sweep:

```text
highest threshold       lowest threshold
bottom-left      ->     top-right
```

Meaning:

```text
High threshold: low FPR, low TPR
Low threshold:  high FPR, high TPR
```

Simple view:

```text
TPR
1.0 |                         low threshold
    |                    *
    |              *
    |        *
0.0 | * high threshold
    └────────────────────────
      0.0                  1.0
              FPR
```

Ideal model rises quickly toward the **top-left**:

```text
high TPR with low FPR
```

---

## 3. PR curve threshold direction

PR axes:

```text
x-axis = Recall
y-axis = Precision
```

Threshold sweep:

```text
highest threshold       lowest threshold
left side        ->     right side
```

Meaning:

```text
High threshold: fewer alerts, usually higher precision, lower recall
Low threshold:  more alerts, usually lower precision, higher recall
```

Simple view:

```text
Precision
1.0 | * high threshold
    |    *
    |       *
    |           *
0.0 |                * low threshold
    └────────────────────────
      0.0                  1.0
             Recall
```

Ideal model stays near the **top-right**:

```text
high recall with high precision
```

---

## 4. One-line summary

```text
High threshold = strict model = fewer fraud alerts
Low threshold  = loose model  = more fraud alerts
```

ROC curve shows:

```text
How much fraud can I catch versus how many normal cases I wrongly flag?
```

PR curve shows:

```text
How much fraud can I catch versus how trustworthy are my fraud alerts?
```

For fraud detection, the **PR curve and threshold table** are usually more practically important than ROC alone.