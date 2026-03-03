from fastapi import FastAPI, Query
from typing import Optional, List
import pandas as pd
import numpy as np

app = FastAPI(title="H!P-Navigator Backend API")

# Load and preprocess data
df = pd.read_csv("hp_master_knowledge_v6.0.csv")
df['semantic_tags'] = df['semantic_tags'].fillna("")

@app.get("/search")
def search(
    q: Optional[str] = Query(None, description="キーワード（曲名・アーティスト名）"),
    tag: Optional[str] = Query(None, description="セマンティックタグ"),
    fame: Optional[str] = Query(None, description="知名度 (standard, hidden, manic)"),
    mood: Optional[str] = Query(None, description="感情スコア (euphoria, sentimental, struggle等)"),
    bpm_min: Optional[int] = Query(None, description="最小BPM"),
    bpm_max: Optional[int] = Query(None, description="最大BPM"),
):
    result_df = df.copy()

    # Keyword search
    if q:
        q_lower = q.lower()
        # Search in title or artist_name
        mask = result_df['title'].str.lower().str.contains(q_lower, na=False) | \
               result_df['artist_name'].str.lower().str.contains(q_lower, na=False)
        result_df = result_df[mask]

    # Tag search
    if tag:
        tag_lower = tag.lower()
        result_df = result_df[result_df['semantic_tags'].str.lower().str.contains(tag_lower, na=False)]

    # Fame search
    if fame:
        if fame == "standard":
            result_df = result_df[result_df['fame_score'] >= 0.3]
        elif fame == "hidden":
            result_df = result_df[(result_df['fame_score'] >= 0.1) & (result_df['fame_score'] < 0.4)]
        elif fame == "manic":
            result_df = result_df[result_df['fame_score'] < 0.1]

    # Mood search (0.8以上のパーセンタイル)
    if mood:
        mood_col = f"score_{mood}"
        if mood_col in df.columns:
            threshold = df[mood_col].quantile(0.8)
            result_df = result_df[result_df[mood_col] >= threshold]

    # BPM search (tempo)
    if bpm_min is not None:
        result_df = result_df[result_df['tempo'] >= bpm_min]
    if bpm_max is not None:
        result_df = result_df[result_df['tempo'] <= bpm_max]

    # Randomly select up to 3 songs
    if len(result_df) == 0:
        return []

    n_samples = min(3, len(result_df))
    sampled_df = result_df.sample(n=n_samples)

    # Replace NaN with None for JSON serialization
    sampled_df = sampled_df.replace({np.nan: None})
    
    # Output all CSV columns as JSON
    records = sampled_df.to_dict(orient="records")
    return records
