# Grammar Scoring Engine for Spoken Audio

Solution for the SHL Hiring Assessment Kaggle competition: predict a continuous MOS-Likert grammar score (0-5) from a 45-60 second speech clip.

## Approach

Grammar depends on both what is said and how it is delivered, so the pipeline combines text and audio signals.

1. **Transcription:** `faster-whisper` (medium) produces transcripts with word timestamps and word confidence. The prompt keeps fillers ("um", "uh") so disfluencies are not cleaned away.
2. **Text and fluency features:** speech rate, pauses, fillers, repetitions, sentence-length statistics, lexical diversity, LanguageTool error rates, DistilGPT-2 perplexity, and sentence embeddings (all-mpnet-base-v2, bge-large-en-v1.5).
3. **Audio features:** layer-wise mean and standard-deviation pooled embeddings from WavLM (base-plus and large), wav2vec2 (base and large) and HuBERT-base. The best layers of each encoder are chosen by cross-validation.
4. **Models:** SVR on audio embeddings, ridge on text embeddings, and tree ensembles (GBM, ExtraTrees, LightGBM) on handcrafted features.
5. **Validation and blending:** repeated stratified 5-fold cross-validation (3 repeats). Out-of-fold predictions are combined with a non-negative linear stacker, evaluated with nested cross-validation. Predictions are clipped to [0, 5].

## Results

| Metric | Value |
|---|---|
| Training RMSE (final model) | 0.295 |
| Cross-validated RMSE | 0.511 |
| Cross-validated Pearson | 0.911 |
| Baseline RMSE (predict the mean) | 1.238 |
| Public leaderboard score | 0.371 |

Audio embeddings were the strongest single views; the blend beats every individual model.
(Update this table if your final model differs.)

## Repository contents

- `grammar_scoring_engine.ipynb`: the full notebook (code, report, plots and outputs)
- `requirements.txt`: Python dependencies
- `README.md`: this file

## How to run

1. Open the notebook on Kaggle, add the competition data, and turn on a GPU and Internet (the pretrained models are downloaded from the Hugging Face hub).
2. Run all cells from top to bottom. The Whisper transcription is the slowest step (about 40 minutes on a GPU) and is cached.
3. The notebook writes `submission.csv` (one row per file in `test.csv`).

## Data

The competition data is **not included** in this repository, as the competition rules do not allow sharing it. Only code and documentation are published here. No external datasets were used; pretrained open-source models are used only as feature extractors.
