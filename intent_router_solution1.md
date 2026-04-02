# Intent Router - Solution 1

## Problem Statement

Build an intent router to classify user queries into **4 intents**:
- `dashboard_load` 
- `semantic_view`
- `map_layer`
- `ask_analyst`

Additionally, extract relevant entities for each intent (e.g., FIELD_NAME, RESERVOIR_ID, COUNTRY_NAME, DATE_RANGE).

---

## Solution 1: BERT Dual-Head Model with Synthetic Data

### Approach
1. **Synthetically generate training data** for all 4 intent classes with labeled entities
2. **Fine-tune BERT using LoRA** with a dual-head architecture:
   - **Head 1 (Classification)**: Intent classification (4-way softmax)
   - **Head 2 (NER)**: Token-level entity extraction (BIO tagging)

### Architecture
```
Input Query → BERT Encoder → [CLS] token → Classification Head → Intent
                          → Token embeddings → NER Head → Entities
```

---

## Pros

- **Single model** handles both tasks - simpler deployment, shared representations
- **End-to-end training** - classification and NER can reinforce each other
- **Control over data distribution** - synthetic data ensures balanced classes
- **Joint optimization** - model learns intent-entity relationships together

---

## Cons

- **Synthetic data quality** - may not capture real-world query variations/noise
- **Domain shift** - model may underperform on production queries that differ from synthetic patterns
- **Data generation effort** - need to create diverse, realistic examples for all entity combinations
- **Overfitting risk** - model may memorize synthetic patterns instead of learning generalizable features
- **Annotation cost** - NER labels require precise token-level annotations in generated data
- **Retraining required** - model needs retraining when query patterns or user behavior changes
- **Schema changes need retraining** - adding new intents or entity types requires regenerating data and retraining

---

## Solution 2: ColBERT + Hierarchical Pipeline with Hybrid NER

### Approach
1. **Generate synthetic data** using LLM (Claude Sonnet 4.5) - ~100 queries per class (400 total)
2. **ColBERT + FAISS k-NN** for route classification
3. **Hierarchical pipeline**: Classification → Route-specific entity targeting → 3-stage NER

### Architecture
```
Query → ColBERT Classification → Route→Entity Map → Hybrid NER (Regex→SpaCy→GLiNER)
```

### Results

| Component | Metric | Value |
|-----------|--------|-------|
| Route Classification | Accuracy | ~90% |
| Entity Extraction | F1 Score | ~70% |
| End-to-End Pipeline | Accuracy | ~63% |
| Latency | Without GLiNER | ~20ms |

---

## Pros

- **Modular design** - components can be updated independently
- **Fast inference** - Regex/SpaCy stages are very fast (<5ms)
- **No joint training** - classification and NER are separate, easier to debug
- **Flexible NER** - 3-stage fallback chain handles different entity types
- **GLiNER fallback** - zero-shot NER covers unknown entities not in lookup tables
- **SpaCy with lookup tables** - EntityRuler enables fast matching of known entities
- **Fuzzy matching** - lightweight addition to handle typos against lookup tables

---

## Cons

- **Error cascade** - wrong classification → wrong entity targets → missed extractions
- **Static lookup tables** - new entities require code changes, doesn't scale
- **Case-sensitivity bug** - SpaCy NER fails on lowercased input
- **No entity disambiguation** - cannot resolve ambiguous entity names
- **GLiNER memory-heavy** - ~3GB, disabled by default

---

## Solution 3: SpaCy + Lookup Table + Fuzzy Matching

### Approach
1. **SpaCy EntityRuler** with lookup tables for known entity extraction
2. **Fuzzy matching** (Levenshtein distance) to handle typos against lookup tables
3. No ML model for NER - purely rule-based

### Architecture
```
Query → Fuzzy Match (typo correction) → SpaCy EntityRuler (lookup tables) → Entities
```

---

## Pros

- **Lightweight** - no heavy ML models, fast inference (<5ms)
- **Interpretable** - exact matching rules, easy to debug
- **No training required** - add entities directly to lookup tables
- **Typo tolerance** - fuzzy matching catches misspellings

---

## Cons

- **Lookup table maintenance** - requires manual updates for new entities, doesn't scale
- **No unknown entity coverage** - misses entities not in lookup tables
- **Rigid patterns** - can't generalize to new entity variations
- **Fuzzy match false positives** - may match unintended similar words
- **No context awareness** - can't disambiguate entities based on surrounding text
