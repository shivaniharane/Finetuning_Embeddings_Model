# 🎯 Fine-Tuning Embedding Models with Sentence Transformers

> **Teaching an AI to better understand the *meaning* of text — by training it on domain-specific question-answer pairs.**

---

## What Is This Project?

This is a **Jupyter Notebook** that takes a pre-trained embedding model (`all-mpnet-base-v2`) and fine-tunes it on a **question-answer dataset** so it becomes better at understanding semantic similarity between questions and their answers.

In plain English:

> You start with an AI that understands *general* English.
> You train it further on *specific* Q&A examples.
> It becomes much better at matching questions to relevant answers.

---

## Why Does This Project Exist?

### The Real-World Scenario

Imagine you're building a **RAG system** for a medical company. You use a general-purpose embedding model to search your document database. A doctor asks:

```
Query:    "What is the recommended dosage for hypertension?"
Document: "For patients with high blood pressure, administer 5mg daily."
```

A **generic** embedding model might give this pair a **low similarity score** because it doesn't strongly associate "hypertension" with "high blood pressure", or "dosage" with "administer Xmg" in a medical context.

A **fine-tuned** embedding model trained on medical Q&A data would score this pair **much higher** — retrieving the right document for the doctor.

This is exactly what this project demonstrates: training the model to understand that a question and its correct answer should have **high similarity**.

---

## The Core Problem: Generic Embeddings vs Fine-Tuned Embeddings

```
┌──────────────────────────────────────────────────────────────────────┐
│            GENERIC EMBEDDING MODEL (Before Fine-Tuning)              │
│                                                                      │
│  Question:  "What is the process of electrolysis?"                   │
│  Answer:    "Electrolysis uses direct electric current to drive      │
│              a non-spontaneous chemical reaction."                   │
│                                                                      │
│  Similarity Score: LOW                                               │
│  ❌ Model doesn't strongly connect the question to its answer        │
│     because it was never specifically trained on Q&A pairs           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│          FINE-TUNED EMBEDDING MODEL (After Fine-Tuning)              │
│                                                                      │
│  Question:  "What is the process of electrolysis?"                   │
│  Answer:    "Electrolysis uses direct electric current to drive      │
│              a non-spontaneous chemical reaction."                   │
│                                                                      │
│  Similarity Score: HIGH ✅                                           │
│  ✅ Model now knows: questions and their correct answers             │
│     should be embedded close together in vector space                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Explained

### 1. What Are Embeddings?

An **embedding** is a list of numbers (a vector) that represents the *meaning* of a piece of text. Similar meanings produce vectors that are mathematically close together.

```
┌──────────────────────────────────────────────────────────────────┐
│                    EMBEDDING SPACE VISUALIZED                    │
│                                                                  │
│   High                                                           │
│   │                                                              │
│   │   ● "car"         ● "automobile"   ← very close             │
│   │   ● "vehicle"                                                │
│   │                                                              │
│   │                    ● "apple"                                 │
│   │                    ● "fruit"       ← close to each other     │
│   │                    ● "banana"                                │
│   │                                                              │
│   └─────────────────────────────────────────────── Low           │
│                                                                  │
│   "car" and "banana" are far apart — different meanings          │
│   "car" and "automobile" are close — same meaning                │
└──────────────────────────────────────────────────────────────────┘

Text  →  Embedding Model  →  [0.12, 0.87, 0.44, 0.63, ...]  (768 numbers)
```

---

### 2. What Is a Sentence Transformer?

A **Sentence Transformer** is a type of neural network built specifically to create high-quality embeddings for *entire sentences* (not just single words). It's built on top of BERT-style transformer architecture.

```
┌───────────────────────────────────────────────────────────────────┐
│               HOW A SENTENCE TRANSFORMER WORKS                    │
│                                                                   │
│  Input sentence:                                                  │
│  "What is the process of electrolysis?"                           │
│          │                                                        │
│          ▼                                                        │
│  ┌───────────────────────────────────┐                            │
│  │        Tokenizer                  │                            │
│  │  Breaks sentence into tokens      │                            │
│  │  ["What", "is", "electrolysis",   │                            │
│  │   "?", ...]                       │                            │
│  └──────────────┬────────────────────┘                            │
│                 │                                                 │
│                 ▼                                                 │
│  ┌───────────────────────────────────┐                            │
│  │     Transformer Encoder           │                            │
│  │  (12 layers of attention)         │                            │
│  │  Understands relationships        │                            │
│  │  between all words                │                            │
│  └──────────────┬────────────────────┘                            │
│                 │                                                 │
│                 ▼                                                 │
│  ┌───────────────────────────────────┐                            │
│  │        Pooling Layer              │                            │
│  │  Compresses all token outputs     │                            │
│  │  into ONE vector for the sentence │                            │
│  └──────────────┬────────────────────┘                            │
│                 │                                                 │
│                 ▼                                                 │
│  [0.12, 0.87, 0.44, 0.63, ...]  ← 768-dimensional vector         │
└───────────────────────────────────────────────────────────────────┘
```

---

### 3. What Is Fine-Tuning?

Fine-tuning is the process of taking a **pre-trained model** (already trained on huge amounts of general text) and continuing its training on a **smaller, specific dataset** to improve performance on a particular task.

```
┌───────────────────────────────────────────────────────────────────┐
│                    THE FINE-TUNING CONCEPT                        │
│                                                                   │
│  Stage 1: Pre-Training (done by researchers, not you)             │
│  ┌───────────────────────────────────────────┐                    │
│  │  Billions of internet text sentences      │                    │
│  │  → Model learns general English language  │                    │
│  │  → Takes weeks on thousands of GPUs       │                    │
│  │  → Result: all-mpnet-base-v2              │                    │
│  └───────────────────────────────────────────┘                    │
│                          │                                        │
│                          ▼                                        │
│  Stage 2: Fine-Tuning (this project — done by you)                │
│  ┌───────────────────────────────────────────┐                    │
│  │  Thousands of Q&A pairs                   │                    │
│  │  → Model learns Q&A similarity patterns   │                    │
│  │  → Takes minutes on one GPU               │                    │
│  │  → Result: finetuned-all-mpnet-base-v2    │                    │
│  └───────────────────────────────────────────┘                    │
│                                                                   │
│  Think of it like:                                                │
│  Pre-training  = Teaching someone to speak English                │
│  Fine-tuning   = Teaching that person to speak medical English    │
└───────────────────────────────────────────────────────────────────┘
```

---

### 4. What Is Cosine Similarity?

Cosine Similarity is the main way to measure how close two vectors (embeddings) are. It measures the **angle** between them, not the distance.

```
┌──────────────────────────────────────────────────────────────────┐
│                 COSINE SIMILARITY EXPLAINED                      │
│                                                                  │
│   Score = +1.0  →  Vectors point in the same direction          │
│                    Sentences are identical in meaning            │
│                                                                  │
│   Score =  0.0  →  Vectors are perpendicular (90°)              │
│                    Sentences are completely unrelated            │
│                                                                  │
│   Score = -1.0  →  Vectors point in opposite directions         │
│                    Sentences have opposite meanings              │
│                                                                  │
│  ┌─────────────────────────────────────────────────┐             │
│  │  Example:                                       │             │
│  │                                                 │             │
│  │  "cat" vs "kitten"    → similarity ≈ 0.91 ✅   │             │
│  │  "cat" vs "dog"       → similarity ≈ 0.73      │             │
│  │  "cat" vs "democracy" → similarity ≈ 0.12 ❌   │             │
│  └─────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

### 5. What Is Cosine Similarity Loss?

This is the **training signal** — the function that tells the model how wrong it was after each prediction, so it can correct itself.

```
┌──────────────────────────────────────────────────────────────────┐
│             HOW COSINE SIMILARITY LOSS WORKS                     │
│                                                                  │
│  Training example:                                               │
│  {                                                               │
│    sentence1: "What is electrolysis?",                           │
│    sentence2: "Electrolysis uses electric current...",           │
│    label:     1.0   ← these should be HIGHLY similar            │
│  }                                                               │
│                                                                  │
│  Step 1: Model encodes both sentences → two vectors              │
│  Step 2: Model computes cosine similarity → e.g., 0.42           │
│  Step 3: Loss = |predicted (0.42) - label (1.0)| = 0.58         │
│  Step 4: Model adjusts weights to reduce this error             │
│  Step 5: Repeat for thousands of examples                        │
│                                                                  │
│  After training:                                                 │
│  Same pair → similarity = 0.89 ✅  (much closer to 1.0)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Full Project Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FULL PROJECT ARCHITECTURE                         │
│                                                                      │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║                    INPUTS                                    ║    │
│  ║                                                              ║    │
│  ║  ┌──────────────────────┐   ┌──────────────────────────┐   ║    │
│  ║  │   Pre-trained Model  │   │   Training Dataset       │   ║    │
│  ║  │  all-mpnet-base-v2   │   │  Mihaiii/qa-assistant    │   ║    │
│  ║  │  (from HuggingFace)  │   │  (from HuggingFace Hub)  │   ║    │
│  ║  └──────────┬───────────┘   └────────────┬─────────────┘   ║    │
│  ╚═════════════╪════════════════════════════╪═════════════════╝    │
│                │                            │                        │
│                ▼                            ▼                        │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║                 PRE-TRAINING EVALUATION                      ║    │
│  ║                                                              ║    │
│  ║  model.similarity(encode(question), encode(answer))         ║    │
│  ║  → Record baseline similarity score                         ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
│                                │                                     │
│                                ▼                                     │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║                    FINE-TUNING                               ║    │
│  ║                                                              ║    │
│  ║  Loss: CosineSimilarityLoss                                  ║    │
│  ║                                                              ║    │
│  ║  For each batch of Q&A pairs:                                ║    │
│  ║   1. Encode question  → vector A                            ║    │
│  ║   2. Encode answer    → vector B                            ║    │
│  ║   3. Compute similarity(A, B)                               ║    │
│  ║   4. Compare to label (1.0 = relevant)                      ║    │
│  ║   5. Compute loss (how wrong the model was)                 ║    │
│  ║   6. Backpropagate → update model weights                   ║    │
│  ║   7. Repeat for 5 epochs                                    ║    │
│  ║                                                              ║    │
│  ║  Evaluate every 100 steps, save checkpoint every 100 steps  ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
│                                │                                     │
│                                ▼                                     │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║                 POST-TRAINING EVALUATION                     ║    │
│  ║                                                              ║    │
│  ║  Same test: model.similarity(encode(question),              ║    │
│  ║                              encode(answer))                ║    │
│  ║  → Compare new score to baseline                            ║    │
│  ║  → Higher score = model improved ✅                         ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
│                                │                                     │
│                                ▼                                     │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║                  OUTPUT                                      ║    │
│  ║                                                              ║    │
│  ║  Saved model: models/finetuned-all-mpnet-base-v2/           ║    │
│  ║  Ready to use in RAG systems for better retrieval           ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Code Walkthrough

### Step 1 — Load the Base Model

```python
model = SentenceTransformer("all-mpnet-base-v2")
```

Downloads and loads the `all-mpnet-base-v2` model from HuggingFace. This model:
- Was pre-trained on over 1 billion sentence pairs
- Produces 768-dimensional vectors
- Achieves strong performance on general semantic similarity tasks

Think of this as your **starting point** — a smart student who knows general English but has never specialized in Q&A.

---

### Step 2 — Baseline Similarity Test

```python
similarity = model.similarity(
    model.encode("What is the process of electrolysis?"),
    model.encode("Electrolysis is a method of using a direct electric current...")
)
print(similarity)
```

This runs **before** training to record a baseline. You encode both sentences into vectors and compute their cosine similarity. The score here is expected to be lower than after fine-tuning.

`model.encode()` converts a text string into its vector representation using the current model weights.

---

### Step 3 — Load the Dataset

```python
dataset = load_dataset("Mihaiii/qa-assistant")
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]
```

Loads the `qa-assistant` dataset from HuggingFace Hub. Each example in this dataset contains:

```
{
  "sentence1": "What is the process of electrolysis?",
  "sentence2": "Electrolysis is a method of using a direct electric current...",
  "label":     1.0    ← means: these two are semantically related
}
```

The `train` split is used to update model weights. The `test` split is used to evaluate performance during training without updating weights.

---

### Step 4 — Define the Loss Function

```python
loss = CosineSimilarityLoss(model)
```

This is the mathematical function that measures **how wrong the model is** during training. For each Q&A pair:

- The model predicts a similarity score between 0 and 1
- The label says what the score *should* be (e.g., 1.0 for a matching pair)
- Loss = difference between prediction and label
- The model adjusts its internal weights to make the loss smaller on the next step

Lower loss = model is getting better at its job.

---

### Step 5 — Configure Training Arguments

```python
args = SentenceTransformerTrainingArguments(
    output_dir="models/finetuned-all-mpnet-base-v2",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=1,
    report_to="none"
)
```

See the full breakdown in the [Training Arguments Explained](#training-arguments-explained) section below.

---

### Step 6 — Train the Model

```python
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss
)

trainer.train()
```

Creates a trainer and starts fine-tuning. The trainer handles:
- Batching the dataset
- Forward pass (making predictions)
- Computing loss
- Backward pass (updating weights via backpropagation)
- Evaluation at checkpoints
- Saving model checkpoints

---

### Step 7 — Post-Training Similarity Test

```python
similarity = model.similarity(
    model.encode("What is the process of electrolysis?"),
    model.encode("Electrolysis is a method of using a direct electric current...")
)
print(similarity)
```

**Identical code to Step 2**, but now run on the fine-tuned model. The similarity score should be noticeably higher, demonstrating that the model improved at recognizing Q&A relevance.

---

## Training Arguments Explained

| Argument | Value | What It Means |
|---|---|---|
| `output_dir` | `models/finetuned-...` | Where to save the trained model and checkpoints |
| `num_train_epochs` | `5` | Train through the entire dataset 5 times |
| `per_device_train_batch_size` | `16` | Process 16 examples at a time during training |
| `per_device_eval_batch_size` | `16` | Process 16 examples at a time during evaluation |
| `warmup_ratio` | `0.1` | Gradually increase learning rate for the first 10% of steps (prevents unstable early training) |
| `fp16` | `True` | Use 16-bit floating point math — 2x faster, uses half the GPU memory |
| `eval_strategy` | `"steps"` | Evaluate on the validation set at regular step intervals |
| `eval_steps` | `100` | Run evaluation every 100 training steps |
| `save_strategy` | `"steps"` | Save a model checkpoint at regular step intervals |
| `save_steps` | `100` | Save a checkpoint every 100 training steps |
| `save_total_limit` | `2` | Keep only the 2 most recent checkpoints to save disk space |
| `logging_steps` | `1` | Log the training loss after every single step |
| `report_to` | `"none"` | Don't send metrics to WandB or TensorBoard |

### What Is an Epoch?

```
Dataset: 1000 training examples
Batch size: 16 examples per batch

1 Epoch = all 1000 examples seen once
        = 1000 / 16 = ~62 steps

5 Epochs = 5 × 62 = ~310 total training steps
```

---

### What Is Warmup?

```
Without warmup:                  With warmup (warmup_ratio=0.1):

Learning rate                    Learning rate
  │                                │         ___________
  │_____________________________   │        /
  │                               │       /
  └──────────────────────── Steps  └──────/─────────────── Steps
                                         warmup
  ← Can cause unstable            ← Gradual ramp-up prevents
    training at start               large erratic weight updates
```

---

## Before vs After Fine-Tuning

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SIMILARITY SCORE COMPARISON                      │
│                                                                      │
│  Test pair:                                                          │
│  Q: "What is the process of electrolysis?"                           │
│  A: "Electrolysis uses direct electric current to drive              │
│      a non-spontaneous chemical reaction."                           │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │  BEFORE fine-tuning (base model):                   │             │
│  │  Similarity = lower score                           │             │
│  │  ████████░░░░░░░░░░░░░░░░  ~0.4–0.6                │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │  AFTER fine-tuning (trained model):                 │             │
│  │  Similarity = higher score ✅                       │             │
│  │  ████████████████████░░░░  ~0.8–0.95               │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  The model learned: questions and their correct answers              │
│  belong close together in vector space.                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Used

**`Mihaiii/qa-assistant`** on HuggingFace Hub

This is a question-answer pair dataset where:
- `sentence1` — a natural language question
- `sentence2` — the correct answer to that question
- `label` — a float (typically 1.0) indicating these two sentences are semantically related

The dataset is pre-split into `train` and `test` sets, making it easy to load and use directly.

Find more datasets suitable for sentence similarity fine-tuning at: [https://huggingface.co/datasets](https://huggingface.co/datasets)

---

## Base Model Used

**`all-mpnet-base-v2`** by Microsoft, hosted on HuggingFace

| Property | Value |
|---|---|
| Architecture | MPNet (Masked and Permuted Pre-training) |
| Embedding Dimension | 768 |
| Max Input Length | 384 tokens |
| Pre-trained on | 1B+ sentence pairs |
| Best use case | Semantic similarity, semantic search, clustering |

Find all available Sentence Transformer models at:
[https://www.sbert.net/docs/sentence_transformer/pretrained_models.html](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

---

## Tools & Libraries

### Core Libraries

| Library | Purpose |
|---|---|
| `sentence-transformers` | The main library — loads, trains, evaluates, and saves embedding models |
| `datasets` | HuggingFace library for loading and managing training datasets |

### Key Classes from `sentence-transformers`

| Class / Function | Purpose |
|---|---|
| `SentenceTransformer` | Loads a pre-trained model and provides `.encode()` and `.similarity()` methods |
| `SentenceTransformerTrainer` | Handles the full training loop — batching, loss, backprop, checkpointing |
| `SentenceTransformerTrainingArguments` | Configuration for the training run (epochs, batch size, save strategy, etc.) |
| `CosineSimilarityLoss` | Loss function that trains the model to match predicted similarity to labeled similarity |

### Key Classes from `datasets`

| Class / Function | Purpose |
|---|---|
| `load_dataset` | Downloads and loads a dataset from HuggingFace Hub by name |

### Infrastructure

| Tool | Purpose |
|---|---|
| **HuggingFace Hub** | Hosts pre-trained models and datasets, downloaded automatically |
| **GPU (Colab T4)** | Required for `fp16=True` training — CPU training is extremely slow |
| **Google Colab** | The recommended runtime environment |

---

## Setup & Installation

### Prerequisites

- **Google Colab** with a **GPU runtime** (free T4 GPU available)
  - Runtime → Change runtime type → Hardware accelerator → **GPU**
- No API keys required — this project is entirely local

### Enable GPU in Colab

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

> ⚠️ `fp16=True` in the training args requires a GPU. On CPU, either set `fp16=False` or training will fail.

### Install Dependencies

```bash
!pip install datasets sentence-transformers
```

That's it — just two libraries.

---

## How to Run

### Run the Full Notebook

Run all cells top to bottom. The notebook will:
1. Load the base model
2. Run the baseline similarity test (record this score)
3. Load the dataset
4. Fine-tune for 5 epochs
5. Run the post-training similarity test
6. Compare scores — the second should be higher

### Use Your Own Dataset

Your dataset must have these three columns:

```python
{
    "sentence1": "Your question here",
    "sentence2": "The corresponding answer here",
    "label":     1.0    # 1.0 = relevant pair, 0.0 = irrelevant pair
}
```

Load it instead of the HuggingFace dataset:

```python
from datasets import Dataset

data = {
    "sentence1": ["Q1", "Q2", "Q3"],
    "sentence2": ["A1", "A2", "A3"],
    "label":     [1.0,  1.0,  1.0]
}

train_dataset = Dataset.from_dict(data)
```

### Use Your Fine-Tuned Model After Training

```python
from sentence_transformers import SentenceTransformer

# Load the saved fine-tuned model
model = SentenceTransformer("models/finetuned-all-mpnet-base-v2")

# Encode and compare
score = model.similarity(
    model.encode("Your question"),
    model.encode("A candidate answer")
)
print(score)
```

---

## Where This Fits in a RAG System

Fine-tuning an embedding model is most valuable when plugged into a **RAG retrieval pipeline**:

```
┌──────────────────────────────────────────────────────────────────────┐
│           FINE-TUNED EMBEDDINGS IN A RAG PIPELINE                    │
│                                                                      │
│  Documents                                                           │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────────────────────────────┐                                │
│  │  Fine-Tuned Embedding Model      │  ← This project's output      │
│  │  (domain-aware vectors)          │                                │
│  └──────────────────┬───────────────┘                                │
│                     │                                                │
│                     ▼                                                │
│              Vector Database                                         │
│              (ChromaDB, Pinecone, etc.)                              │
│                     │                                                │
│  User Query         │                                                │
│      │              │                                                │
│      ▼              │                                                │
│  Fine-Tuned Embedding Model (same model)                             │
│      │                                                               │
│      ▼                                                               │
│  Similarity Search in Vector DB                                      │
│      │                                                               │
│      ▼                                                               │
│  More relevant chunks retrieved ✅                                   │
│      │                                                               │
│      ▼                                                               │
│  LLM generates better answers                                        │
└──────────────────────────────────────────────────────────────────────┘

Generic embedding model  →  retrieves okay results
Fine-tuned embedding     →  retrieves domain-relevant results ✅
```

---

## Glossary for Beginners

| Term | Plain English Explanation |
|---|---|
| **Embedding** | A list of numbers representing the meaning of a piece of text |
| **Vector** | Another word for embedding — a list of numbers |
| **Embedding Dimension** | How many numbers are in each vector (here: 768) |
| **Sentence Transformer** | A neural network that produces embeddings for full sentences |
| **Pre-trained Model** | A model already trained on huge data by researchers — your starting point |
| **Fine-Tuning** | Further training a pre-trained model on your specific data to improve performance |
| **Cosine Similarity** | A score (−1 to 1) measuring how similar two vectors are in direction |
| **Loss Function** | A formula that measures how wrong the model's predictions are |
| **CosineSimilarityLoss** | Loss that teaches the model to match its similarity predictions to given labels |
| **Backpropagation** | The algorithm that adjusts model weights based on the loss — how learning happens |
| **Epoch** | One full pass through the entire training dataset |
| **Batch Size** | How many examples the model sees before updating its weights |
| **Warmup** | Gradually increasing the learning rate at the start of training for stability |
| **Learning Rate** | How big a step the model takes when updating weights — too high = unstable |
| **FP16** | 16-bit floating point precision — uses half the GPU memory, trains 2x faster |
| **Checkpoint** | A snapshot of the model saved during training so you can resume or compare |
| **HuggingFace Hub** | An online repository hosting thousands of pre-trained models and datasets |
| **`model.encode()`** | Converts a text string into its embedding vector using the model |
| **`model.similarity()`** | Computes cosine similarity between two embedding vectors |
| **Semantic Similarity** | How similar two texts are in *meaning* (not just word overlap) |
| **Domain-Specific** | Specialized to a particular field (e.g., medical, legal, Q&A) |
| **Baseline** | The model's performance score *before* any fine-tuning — used for comparison |
