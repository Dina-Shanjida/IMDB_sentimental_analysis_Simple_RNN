# 🎬 IMDb Sentiment Analysis using Simple RNN (PyTorch)

This project is a **Natural Language Processing (NLP)** model that performs **sentiment classification** on movie reviews using a **Simple Recurrent Neural Network (RNN)** built with PyTorch.

The model predicts whether a movie review is **positive or negative** based on text input.

---

## 📊 Dataset

- Dataset used: IMDb Movie Reviews Dataset (Kaggle 50k labeled sentiment dataset)  
- Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  
- Total data: 50,000 movie reviews  
- Labels:
  - `positive` → 1
  - `negative` → 0  

For this project, a **random sample of 20,000 reviews** was used to reduce training time.

---

## 🧠 Project Workflow

### 1. Data Loading
- Dataset loaded using `kagglehub`
- Converted into Pandas DataFrame

### 2. Preprocessing
- Lowercasing text
- Removing punctuation and special characters
- Tokenization (splitting text into words)

### 3. Vocabulary Building
- Built word-to-index mapping (vocab)
- Rare words filtered (`frequency > 2`)
- `<UNK>` token used for unknown words

### 4. Text Encoding
- Each review converted into numerical indices
- Sequence length limited to `MAX_LEN = 200`

### 5. Dataset Preparation
- Train / Validation / Test split (70/15/15)
- Custom PyTorch Dataset class used
- Padding applied using `pad_sequence`

### 6. Model Architecture
- Simple RNN model using PyTorch:
  - Embedding Layer
  - RNN Layer (tanh activation)
  - Fully Connected Layer (Binary output)

### 7. Training
- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Gradient Clipping applied (`clip_grad_norm_`)
- Trained for 30 epochs on GPU (CUDA)

### 8. Evaluation
- Accuracy calculated on:
  - Train set
  - Validation set
  - Test set

### 9. Prediction
- Custom function to predict sentiment from raw text input

---

## 🛠️ Technologies Used

- Python 🐍
- PyTorch 🔥
- Pandas 📊
- Scikit-learn 🤖
- KaggleHub 📦
- Regular Expressions (re)

---

## 📚 Key Learnings

- NLP preprocessing pipeline from scratch  
- Tokenization and vocabulary building  
- Text → numerical representation conversion  
- Embedding layer usage in deep learning  
- Sequence modeling using RNN  
- Importance of padding and fixed sequence length  
- Training deep learning models on GPU (CUDA)  
- Gradient clipping for stable RNN training  
- End-to-end ML pipeline in PyTorch  

---

## 📈 Model Performance

| Dataset | Accuracy |
|--------|----------|
| Train | ~73-74% |
| Validation | ~53-54% |
| Test | ~52–53% |

⚠️ Note: Simple RNN has limitations in capturing long-term dependencies, which affects performance.

---

## ⚠️ Limitations

- Simple RNN struggles with long text dependencies  
- No contextual understanding like Transformers  
- Performance is limited due to architecture simplicity  
- No pretrained embeddings used (like Word2Vec/GloVe)

---

## 🚀 Future Improvements

- Replace RNN with LSTM / GRU  
- Use Bidirectional LSTM  
- Add Dropout layers  
- Use pretrained embeddings (GloVe / Word2Vec)  
- Use Transformer models (BERT, RoBERTa)  
- Better hyperparameter tuning  
- Train on full 50K dataset  
- Add precision, recall, F1-score evaluation  

---

## 📌 Example Predictions

```python
predict("This movie was absolutely amazing")
# Output: positive

predict("Worst movie I have ever seen")
# Output: negative
