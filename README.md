# Identifying Healthcare Entities using Named Entity Recognition (NER) and Conditional Random Fields (CRF) under Syntactic processing

## Overview

This project implements a **custom Named Entity Recognition (NER)** pipeline for identifying **diseases** and **treatments** in healthcare text. It uses **spaCy** for part-of-speech (POS) tagging and **Conditional Random Fields (CRF)** for token-level sequence labeling. Predicted entities are then grouped into a disease-to-treatment mapping.

The implementation is provided in the Jupyter notebook [`SurendranathDuvvuru_SyntacticProcessing_Assignment.ipynb`](SurendranathDuvvuru_SyntacticProcessing_Assignment.ipynb).

## Objectives

- Reconstruct sentences and corresponding token labels from word-per-line input files.
- Explore frequently occurring nouns and proper nouns in the corpus.
- Engineer contextual and linguistic features for CRF-based sequence labeling.
- Train a CRF model to identify diseases and treatments.
- Evaluate predictions using weighted F1 score.
- Extract disease–treatment associations from the labeled test sentences.

## Tech Stack

- **Python** and **Jupyter Notebook / Google Colab**
- **spaCy** (`en_core_web_sm`) for POS tagging
- **sklearn-crfsuite** for CRF modeling and evaluation
- **collections.Counter** for word-frequency analysis

## Dataset

The notebook expects four text files:

| File | Purpose |
| --- | --- |
| `train_sent` | Training text, one token per line |
| `train_label` | Corresponding training labels |
| `test_sent` | Test text, one token per line |
| `test_label` | Corresponding test labels |

Blank lines separate sentences. The preprocessing function combines the tokens between blank lines into complete sentences and similarly reconstructs label sequences.

The entity labels used in the notebook are:

| Label | Meaning |
| --- | --- |
| `D` | Disease |
| `T` | Treatment |
| `O` / `0` | Other / non-entity token (verify encoding in your dataset) |

**Note:** The extraction logic explicitly skips the string `'0'`; check the dataset's actual non-entity label before running or adapting it. The four dataset files are **not embedded in the notebook** and must be supplied separately.

## Project Workflow

### 1. Data preprocessing

The `process_data_file()` function reads the word-per-line files and reconstructs sentences and label sequences. The notebook then displays sample sentences and counts training and test sequences.

### 2. Concept identification

spaCy POS tags are used to collect tokens tagged `NOUN` or `PROPN`. `Counter` identifies the 25 most frequent such tokens across training and test sentences.

### 3. Feature engineering

The `getFeaturesForOneWord()` function builds token features, including:

- Lowercased token text
- Last two and three characters
- Uppercase, digit, and initial-capital indicators
- Current token POS tag
- Previous-token lexical, capitalization, numeric, and POS features
- Beginning/end-of-sentence markers

`getFeaturesForOneSentence()` constructs feature sequences, and `getLablesInListForOneSentence()` converts label strings into token-label lists.

### 4. CRF model training

The model is trained using:

```python
crf = sklearn_crfsuite.CRF(max_iterations=100)
crf.fit(X_train, Y_train)
```

### 5. Evaluation

The notebook predicts labels for test sentences and calculates the **weighted token-level F1 score**:

```python
Y_pred = crf.predict(X_test)
metrics.flat_f1_score(Y_test, Y_pred, average="weighted")
```

The notebook computes this metric, but this README does not claim a numeric result without verified execution output.

### 6. Disease–treatment extraction

For each test sentence, predicted `D` tokens are concatenated into a disease phrase and predicted `T` tokens into a treatment phrase. When both are present, the association is stored in a dictionary:

```python
{
    "disease name": ["treatment phrase"]
}
```

Multiple treatment phrases can be appended under the same disease key. This is a **sentence-level heuristic**, not a separately trained relation-extraction model; it may combine unrelated entities when multiple diseases or treatments appear in one sentence.

The assignment also mentions **hereditary retinoblastoma**. The final notebook cells display the extracted dictionary and its first entry; they do **not** explicitly look up that disease by name.

## Limitations and Future Improvements

- Ensure spaCy tokenization aligns with the whitespace-tokenized labels before training; mismatches can cause incorrect feature alignment.
- Verify that every sentence has the same number of tokens and labels.
- Handle a final sentence even if the input file has no trailing blank line.
- Consider entity-level precision, recall, and F1 alongside weighted token-level F1.
- Improve extraction for sentences containing multiple diseases and treatments.
- Add an explicit lookup for a requested disease, for example:

  ```python
  diseases_and_treatments.get("hereditary retinoblastoma", [])
  ```

## Author

**Surendranath Duvvuru**  
GitHub: [SurendranathDuvvuru](https://github.com/SurendranathDuvvuru)

## Disclaimer

This project is an educational NLP exercise. Extracted disease–treatment associations are predictions from text and **must not be interpreted as medical recommendations**.
