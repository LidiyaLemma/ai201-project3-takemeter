# TakeMeter Planning Document
## Project: Discourse Quality Classifier for r/LetsTalkMusic

---

## 1. Community

**Chosen community:** r/LetsTalkMusic

r/LetsTalkMusic is a music discussion subreddit explicitly designed for substantive conversation about artists, albums, and songs across all genres. The discourse varies enormously — from deeply reasoned musical analysis to one-line reactions — making it a strong candidate for a text classifier.

---

## 2. Labels

### `analytical`
Post makes a structured argument about music using specific reasoning — referencing production, songwriting, vocals, instrumentation, or influences with supporting detail.

### `hot_take`
Bold, confident claim about an artist, album, or song without providing specific supporting evidence.

### `contextual`
Post centers on a personal story or emotional connection to music rather than a musical argument.

### `low_effort`
Minimal substance — one-liners, vague reactions, or simple questions.

---

## 3. Hard Edge Cases

**analytical vs contextual:** If the post's primary substance is a personal story and the musical observation is vague, label it contextual. If the post primarily argues something about the music itself, label it analytical.

**hot_take vs analytical:** If the post provides specific verifiable musical detail, label it analytical. If the reasoning is vague or decorative, label it hot_take.

---

## 4. Data Collection Plan

- Source: r/LetsTalkMusic weekly threads and top posts
- Target: 50 examples per label (200 total)
- If a label is underrepresented, supplement from r/popheads or r/indieheads

---

## 5. Evaluation Metrics

- Overall accuracy
- Per-class F1 score
- Confusion matrix
- Macro-averaged F1

Accuracy alone is not enough because if low_effort dominates the dataset, a model predicting only that label would score well but be useless.

---

## 6. Definition of Success

- Overall accuracy ≥ 70% on test set
- F1 score ≥ 0.65 for every label
- Fine-tuned model outperforms Groq baseline by at least 10 points in macro F1

---

## 7. AI Tool Plan

- Used Claude to stress-test label definitions and generate edge cases
- Manually labeled all 200 examples with AI used as a secondary check
- Used Claude to identify patterns in wrong predictions after evaluation
