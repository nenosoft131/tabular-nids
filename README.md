# 📊 Representation Learning & Downstream Evaluation Framework

This repository provides a unified framework for **embedding generation**, **end-to-end modeling**, and **downstream evaluation** for tabular and textual data. It combines multiple **self-supervised learning (SSL)** approaches with downstream tasks to evaluate the quality and usefulness of learned representations.

---

## 🚀 Key Features

* Multiple embedding generation techniques (SSL + deep learning)
* End-to-end model training pipelines
* Downstream evaluation (classification & analysis)
* Embedding visualization and clustering metrics
* Modular and extensible project structure

---

## 📁 Project Structure

```
.
├── SCARF/
├── TabICL/
├── Tabular-Class-Conditioned-SSL/
│   └── job.sh                  # Embedding generation script

├── AutoEncoder/
├── Bert/
│   └── run.py                  # End-to-end training + embeddings

├── Downstream_Models/
├── OVR/
├── Visualisation_SI_DBI/
```

---

## 🔹 Embedding Generation Modules

These modules generate embeddings using self-supervised or contrastive learning techniques:

### SCARF

* Self-Supervised Contrastive Learning for tabular data
* Learns robust and invariant feature representations

### TabICL

* In-context learning approach for tabular datasets
* Captures contextual relationships between samples

### Tabular-Class-Conditioned-SSL

* Class-conditioned self-supervised learning
* Incorporates label-aware structure into embeddings

### ▶️ Run Embedding Generation

```bash
cd <module_folder>
bash job.sh
```

---

## 🔹 End-to-End Models

### AutoEncoder

* Unsupervised neural network
* Learns compressed latent representations via reconstruction

### Bert

* Transformer-based contextual embedding model
* Suitable for text or hybrid tabular-text data

### ▶️ Run End-to-End Models

```bash
cd <module_folder>
python run.py
```

---

## 🔹 Downstream Evaluation

### Downstream_Models

* Train ML models on generated embeddings
* Examples: Logistic Regression, MLP, Random Forest

### OVR (One-vs-Rest)

* Multi-class classification using binary classifiers
* Useful for handling class imbalance

---

## 🔹 Visualization & Metrics

### Visualisation_SI_DBI

Evaluate embedding quality using:

* **Silhouette Index (SI)** → Measures cluster cohesion
* **Davies-Bouldin Index (DBI)** → Measures cluster separation

### ▶️ Run Visualization

```bash
cd Visualisation_SI_DBI
python visualize.py
```

---

## 🔄 Workflow

1. **Generate embeddings**

   * Use one of:

     * `SCARF/`
     * `TabICL/`
     * `Tabular-Class-Conditioned-SSL/`
     * `AutoEncoder/`
     * `Bert/`

2. **Train downstream models**

   * Use:

     * `Downstream_Models/`
     * `OVR/`

3. **Evaluate and visualize**

   * Use:

     * `Visualisation_SI_DBI/`

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch or TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Matplotlib / Seaborn



## ▶️ Example Pipeline

```bash
# Step 1: Generate embeddings
cd SCARF
bash job.sh

# Step 2: Train downstream model
cd ../Downstream_Models
python train.py

# Step 3: Visualize embeddings
cd ../Visualisation_SI_DBI
python visualize.py
```

---

## 📌 Notes

* Each module is independent and can be executed separately
* Ensure consistent preprocessing across all modules
* Store embeddings in a common format (e.g., `.npy`, `.csv`)
* Update file paths and configurations as required

---

## 📈 Future Work

* Unified pipeline script for automation
* Hyperparameter tuning integration
* Benchmark dataset support
* Automated evaluation dashboards

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit a pull request.

