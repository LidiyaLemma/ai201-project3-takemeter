# TakeMeter — Discourse Quality Classifier for r/LetsTalkMusic

## Community
r/LetsTalkMusic is a music discussion subreddit designed for substantive conversation about artists, albums, and songs. Discourse quality varies enormously — from deeply reasoned musical analysis to one-line reactions — making it a strong candidate for a text classifier.

## Labels
| Label | Definition |
|---|---|
| **analytical** | Post makes a structured argument about music using specific reasoning — referencing production, songwriting, vocals, or influences with supporting detail |
| **hot_take** | Bold, confident claim about an artist or album without specific supporting evidence |
| **contextual** | Post centers on a personal story or emotional connection to music rather than a musical argument |
| **low_effort** | Minimal substance — one-liners, vague reactions, or simple questions |

## Data Collection
- **Source:** r/LetsTalkMusic (posts and comments)
- **Threads used:** WHYBLT weekly threads, General Discussion threads, top posts
- **Total examples:** 200
- **Labeling:** Manual annotation by the author

### Label Distribution
| Label | Count |
|---|---|
| low_effort | 74 |
| contextual | 47 |
| analytical | 40 |
| hot_take | 39 |

### Difficult Examples
1. **"Every song from the Kelela album has been phenomenal — it scratches that 90s soul ballad itch while staying contemporary."** — Could be `analytical` (mentions a musical quality) or `hot_take` (bold claim with minimal argument). Labeled `hot_take` because "scratches that itch" is a feeling, not a musical argument.

2. **"All Things Must Pass takes you on the journey of a child growing up... you can hear this in Isn't It a Pity where the song feels as though it is searching for understanding."** — Could be `contextual` (personal interpretation) or `analytical` (discusses the album's meaning with specific song references). Labeled `analytical` because it makes a structured argument about the music itself.

3. **"I've been to hundreds of shows in my life. I'd much rather go to a 500-1000 person venue and be only 20 feet from the artist while paying $20-30."** — Could be `contextual` (personal experience) or `low_effort` (no musical argument). Labeled `contextual` because it shares a specific personal experience with detail.

## Fine-Tuning Pipeline
- **Base model:** distilbert-base-uncased
- **Training approach:** Fine-tuned for sequence classification with a 4-label classification head
- **Train/val/test split:** 70% / 15% / 15% (stratified)
- **Key hyperparameter decisions:**
  - `num_train_epochs = 3` — standard for small datasets; more epochs risk overfitting on 200 examples
  - `learning_rate = 2e-5` — standard starting point for BERT-family fine-tuning
  - `per_device_train_batch_size = 16` — fits T4 GPU comfortably

## Evaluation Report

### Overall Accuracy
| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **63.3%** |
| Fine-tuned DistilBERT | **46.7%** |

### Per-class Metrics (Fine-tuned Model)
| Label | Precision | Recall | F1 |
|---|---|---|---|
| analytical | 0.00 | 0.00 | 0.00 |
| hot_take | 0.00 | 0.00 | 0.00 |
| contextual | 0.60 | 0.43 | 0.50 |
| low_effort | 0.44 | 1.00 | 0.61 |

### Per-class Metrics (Baseline)
| Label | Precision | Recall | F1 |
|---|---|---|---|
| analytical | 0.50 | 0.17 | 0.25 |
| hot_take | 0.43 | 0.50 | 0.46 |
| contextual | 0.78 | 1.00 | 0.88 |
| low_effort | 0.67 | 0.73 | 0.70 |

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Wrong Predictions Analysis
1. **True: analytical → Predicted: low_effort** — "The dynamic between Amy Lee's vocals and the metal instrumentation is a back and forth that just works." The model missed this because analytical posts can be short and the model over-predicted low_effort.

2. **True: hot_take → Predicted: low_effort** — "Talk Talk are amazing. They do not have one bad song in their catalog." Short hot_takes look identical to low_effort posts in length, confusing the model.

3. **True: contextual → Predicted: low_effort** — "I was stationed at Eielson AFB and heard Los Lobos for the first time." Personal stories without musical vocabulary got misclassified as low effort.

### What the Model Learned vs. What I Intended
The fine-tuned model learned one thing: predict `low_effort`. Because `low_effort` had 74 examples (37% of the data) versus only 39 `hot_take` examples, the model optimized for the most common label. It achieved 47% accuracy by predicting `low_effort` for almost everything.

What I intended was a model that could distinguish substantive musical reasoning (`analytical`) from bold unsupported claims (`hot_take`) from personal stories (`contextual`) from noise (`low_effort`). The Groq zero-shot baseline did this far better (63%) because it could reason from label definitions alone without needing training examples.

The key lesson: with only 200 examples and an imbalanced label distribution, fine-tuning a small model does not outperform a large zero-shot model.

## AI Tool Usage
- Claude (claude.ai) was used to assist with label design, data labeling, and planning.md structure
- All 200 examples were collected from r/LetsTalkMusic and reviewed by the author
- Claude was used to help write this README based on actual experimental results
