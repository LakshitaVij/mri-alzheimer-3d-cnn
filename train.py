# ==========================
# IMPORTS
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim #optimizer (how weights get updated)
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split #split data into train/val/test
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix #measure performance

from model import Simple3DCNN
from dataset import BrainDataset, files


# ==========================
# TRAIN / VAL / TEST SPLIT
# ==========================
train_files, temp_files = train_test_split(
    files,
    test_size=0.2,
    stratify=[label for _, label in files],
    random_state=42
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    stratify=[label for _, label in temp_files],
    random_state=42
)

print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Test: {len(test_files)}")


# ==========================
# DATA LOADERS
# ==========================
train_dataset = BrainDataset(train_files)
val_dataset   = BrainDataset(val_files)
test_dataset  = BrainDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)


# ==========================
# MODEL SETUP
# ==========================
model = Simple3DCNN()

# class weighting
labels = [label for _, label in train_files] #extract only the labels in the train files
pos_weight_value = (labels.count(0) / labels.count(1)) #IMPORTANT! It calculates how much rarer dementia cases are so we can give them more importance during training.
pos_weight = torch.tensor([pos_weight_value]) #make it into a tensor

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


# ==========================
# TRAINING LOOP (EARLY STOP)
# ==========================
"""
epochs = 15
best_val_loss = float("inf")
patience = 3 #How many epochs am I willing to wait without improvement before stopping?
counter = 0

for epoch in range(epochs):

    # ---- TRAIN ----
    model.train()
    total_loss = 0

    for volumes, labels in train_loader:
        labels = labels.unsqueeze(1).float()

        outputs = model(volumes)
        loss = criterion(outputs, labels)

        optimizer.zero_grad() #“Reset all stored gradients to zero before computing new ones.”
        loss.backward() #“Compute how each weight contributed to the loss.”
        optimizer.step() #“Update each weight using its gradient.”

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for volumes, labels in val_loader: #matching shape expected by BCEWithLogitsLoss
            labels = labels.unsqueeze(1).float()

            outputs = model(volumes) 
            loss = criterion(outputs, labels)

            val_loss += loss.item() #loss is a tensor. item() turns it into a normal number. We add it to val_loss to accumulate total validation error.

    val_loss /= len(val_loader) # We divide by number of batches bc average validation loss, not total.

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break


print("Best model saved as best_model.pt")
"""



model = Simple3DCNN()                         # recreate architecture
model.load_state_dict(torch.load("best_model.pt", weights_only=True))
model.eval()  

# ==========================
# EVALUATION FUNCTION
# ==========================
def evaluate(model, loader):
 
    model.eval() #Switches model to evaluation mode
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad(): #Turns off gradient tracking
        for volumes, labels in loader:
            labels = labels.unsqueeze(1).float()

            outputs = model(volumes) #Run forward pass → get logits.
            probs = torch.sigmoid(outputs) #Convert logits → probabilities (0 to 1).
            preds = (probs > 0.5).float() #If probability > 0.5 → predict 1 Else → predict 0

            all_probs.extend(probs.view(-1).cpu().numpy()) #flattens the predictions into a 1D list, moves them from GPU to CPU, and converts them into NumPy format so scikit-learn can compute metrics.
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, auc, cm


# ==========================
# LOAD BEST MODEL + FINAL EVAL
# ==========================
model.load_state_dict(torch.load("best_model.pt"))

train_acc, train_auc, _ = evaluate(model, train_loader)
val_acc, val_auc, _ = evaluate(model, val_loader)
test_acc, test_auc, test_cm = evaluate(model, test_loader)

print("\nFINAL RESULTS")
print("Train Acc:", train_acc, "AUC:", train_auc)
print("Val   Acc:", val_acc,   "AUC:", val_auc)
print("Test  Acc:", test_acc,  "AUC:", test_auc)

print("\nTest Confusion Matrix:")
print(test_cm)