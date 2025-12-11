import os
import pandas as pd
import pickle
from tqdm import tqdm
from typing import List
import math
from app.services.search_service import clean_address

USE_TRANSFORMERS = True

if USE_TRANSFORMERS:
    try:
        from transformers import pipeline
    except Exception as e:
        print("transformers not available:", e)
        USE_TRANSFORMERS = False

# ---------- CONFIG ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FINAL_CSV = os.path.join(PROJECT_ROOT, "final_with.csv")  # your full raw dataset
PLACES_PICKLE_IN = os.path.join(DATA_DIR, "places_index_with_categories.pkl") # existing index with embeddings AND categories
PLACES_PICKLE_OUT = os.path.join(DATA_DIR, "places_index_with_reviews.pkl")  # output
MAX_REVIEWS_PER_PLACE = 10
SENTIMENT_BATCH_SIZE = 16  # tune for memory
# ----------------------------

def load_datasets():
    print("Loading CSV:", FINAL_CSV)
    df = pd.read_csv(FINAL_CSV, sep=",", quotechar='"', escapechar="\\", engine="python")
    print("Rows in final CSV:", len(df))
    print("Loading places index pickle:", PLACES_PICKLE_IN)
    with open(PLACES_PICKLE_IN, "rb") as f:
        places_df = pickle.load(f)
    print("Loaded places index rows:", len(places_df))
    return df, places_df

def aggregate_reviews(df: pd.DataFrame, max_reviews: int = 10):
    """
    Group review_text by place_id and return a mapping: place_id -> list[str]
    We assume df has 'place_id', 'review_text' and 'review_datetime' (optionally).
    We keep most recent reviews if review_datetime exists.
    """
    # ensure columns exist
    if "place_id" not in df.columns or "review_text" not in df.columns:
        raise ValueError("final_with.csv must contain 'place_id' and 'review_text' columns")

    # if review_datetime exists, parse and sort
    if "review_datetime" in df.columns:
        df["review_datetime_parsed"] = pd.to_datetime(df["review_datetime"], errors="coerce")
        df_sorted = df.sort_values("review_datetime_parsed", ascending=False)
    else:
        df_sorted = df.copy()

    grouped = {}
    for place_id, group in tqdm(df_sorted.groupby("place_id"), desc="grouping places"):
        texts = []
        for t in group["review_text"].dropna().astype(str).values:
            t = t.strip()
            if not t:
                continue
            texts.append(t)
            if len(texts) >= max_reviews:
                break
        grouped[place_id] = texts
    return grouped

def prepare_sentiment_pipeline():
    if not USE_TRANSFORMERS:
        return None
    # Choose a multilingual sentiment model with reasonable speed/quality.
    # 'nlptown/bert-base-multilingual-uncased-sentiment' returns 1..5 star labels; it's fast enough and multilingual.
    try:
        pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)
    except Exception:
        # fallback to a different model
        pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=-1)
    return pipe

def score_reviews_sentiment(pipe, texts: List[str]) -> float:
    """
    Given a list of short texts, returns a normalized sentiment score in [0..1].
    If pipe is None, returns None.
    """
    if not texts:
        return None
    if pipe is None:
        return None

    scores = []
    # batch
    for i in range(0, len(texts), SENTIMENT_BATCH_SIZE):
        batch = texts[i : i + SENTIMENT_BATCH_SIZE]
        preds = pipe(batch, truncation=True)
        for p in preds:
            lab = p.get("label", "")
            # try parse a numeric label (nlptown -> '1 star'..'5 stars')
            import re
            m = re.search(r"(\d)", str(lab))
            if m:
                val = int(m.group(1))
                scores.append((val - 1) / 4.0)  # 1..5 -> 0..1
                continue
            # otherwise map POSITIVE/NEGATIVE or probability
            if "POSITIVE" in lab.upper():
                scores.append(float(p.get("score", 1.0)))
            elif "NEGATIVE" in lab.upper():
                scores.append(1.0 - float(p.get("score", 1.0)))
            else:
                scores.append(float(p.get("score", 0.5)))
    if not scores:
        return None
    return float(sum(scores) / len(scores))

def main():
    raw_df, places_df = load_datasets()

    print("Aggregating reviews per place (max %d)... " % MAX_REVIEWS_PER_PLACE)
    reviews_map = aggregate_reviews(raw_df, MAX_REVIEWS_PER_PLACE)

    print("Preparing sentiment pipeline (this may download a model the first time)...")
    pipe = None
    if USE_TRANSFORMERS:
        pipe = prepare_sentiment_pipeline()
        if pipe is None:
            print("Warning: sentiment pipeline not available, sentiment will be left empty.")
        else:
            print("Sentiment pipeline ready.")

    # For each place in places_df, add 'reviews' (list) and compute 'sentiment' numeric (0..1) optionally
    places_df = places_df.copy()
    # Make sure index is range
    if "place_id" not in places_df.columns:
        raise ValueError("places_index.pkl must contain 'place_id' column.")

    reviews_col = []
    sentiment_col = []
    sentiment_label_col = []

    place_count_with_reviews = 0
    place_count_with_sentiment = 0

    for idx, row in tqdm(places_df.iterrows(), total=len(places_df), desc="processing places"):
        pid = row["place_id"]
        texts = reviews_map.get(pid, [])
        if texts:
            place_count_with_reviews += 1
        reviews_col.append(texts)

        # compute sentiment
        sent_score = None
        sent_label = None
        if texts and pipe is not None:
            try:
                sent_score = score_reviews_sentiment(pipe, texts)
                if sent_score is not None:
                    place_count_with_sentiment += 1
                    if sent_score >= 0.75:
                        sent_label = "positive"
                    elif sent_score >= 0.45:
                        sent_label = "mixed"
                    else:
                        sent_label = "negative"
            except Exception as e:
                # do not fail whole run for a few errors
                print("Sentiment scoring failed for", pid, ":", repr(e))
                sent_score = None
        sentiment_col.append(sent_score)
        sentiment_label_col.append(sent_label)

    # assign columns
    places_df["reviews"] = reviews_col
    places_df["sentiment"] = sentiment_col
    places_df["sentiment_label"] = sentiment_label_col
    places_df["address"] = places_df["address"].apply(lambda x: clean_address(x, transliterate=False))

    print(f"Places with reviews: {place_count_with_reviews}/{len(places_df)}")
    print(f"Places with sentiment computed: {place_count_with_sentiment}/{len(places_df)}")

    # Save pickle
    out_path = PLACES_PICKLE_OUT
    print("Saving updated places index to:", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(places_df, f)

    print("Done. You can now point loader to the new pickle file (places_index_with_reviews.pkl).")

if __name__ == "__main__":
    main()
