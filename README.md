# Digit Difference Predictor with Multi-Head Attention

This project implements a neural network using **multi-head attention** to solve a synthetic sequence task:

> Given an input sequence `XY[0-5]+`, where `X` and `Y` are two digits, count how many times `X` and `Y` appear in the rest of the sequence, and return `#X - #Y`.

---

## 🧠 Example

```text
Input: 1213211
       | | └──── tail of the sequence
       X Y
Tail:        3 2 1 1
Count of 1:  3
Count of 2:  1
Output:      2  (3 - 1)
```

---

## 📦 Features

- Implements **Multi-Head Attention** from _"Attention Is All You Need"_
- Learnable **positional encodings**
- Pure **PyTorch** implementation
- Includes two attention variants:
  - Standard multi-head attention
  - Optimized merged-head version
- Achieves **100% training accuracy**
- Contains detailed inline explanations

---

## 🛠 Requirements

- Python 3.7+
- PyTorch 1.10+

Install dependencies:

```bash
pip install torch
```

---

## 🏗 Model Architecture

- **Embedding Layer** – Maps tokens to vectors
- **Positional Encoding** – Learnable encoding for token positions
- **Encoding Layers** – Stack of multi-head self-attention + feed-forward
- **Decoding Layer** – Queries final representation using learned query vector
- **Fully Connected Layer** – Predicts the result: shifted `#X - #Y`

---

## 🚀 Training

Training is performed on synthetic data generated on the fly. Run the script with:

```bash
python3 train.py
```

During training, the script prints:

- Step number
- Loss
- Accuracy

Example:

```text
[0/24999] loss: 1.808, accuracy: 0.094
...
Final accuracy: 1.000, expected 1.000
```

---

## 🧪 Attention Variants

### 1. `MultiHeadAttention`
- Implements traditional multi-head attention
- Separate scaled dot-product attention per head
- Final output projection

### 2. `MultiHeadAttentionMerged`
- Uses a single scaled dot-product attention
- Merges head computations for efficiency
- Faster training and lower memory footprint

---

## 📂 File Structure

- `Net` – Main network model
- `get_data_sample()` – Synthetic data generator
- `Attention`, `MultiHeadAttention`, `MultiHeadAttentionMerged` – Attention modules
- `EncodingLayer` – Encoder block
- Training loop at the end of the script

---

## 📘 Notes

- Loss function: `CrossEntropyLoss`
- Labels are shifted by `SEQ_LEN` to avoid negative values
- The model is self-contained and does not require an external dataset

---

## 📄 License

This project is provided for educational purposes. No license specified.
