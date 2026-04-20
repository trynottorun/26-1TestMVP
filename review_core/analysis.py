from collections import Counter
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from review_core.shared import clean_review_text_for_analysis, safe_to_datetime, score_to_label, tokenize_review_text


WORDCLOUD_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def resolve_wordcloud_font() -> Optional[str]:
    for font_path in WORDCLOUD_FONT_CANDIDATES:
        if Path(font_path).exists():
            return font_path
    return None


def normalize_analysis_dataframe(dataframe: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if dataframe is None or dataframe.empty:
        return dataframe
    normalized_df = dataframe.copy()
    column_mapping = {
        "작성자명": "author",
        "리뷰 내용": "review",
        "평점": "score",
        "작성일": "date",
        "앱 버전": "app_version",
    }
    normalized_df = normalized_df.rename(columns=column_mapping)
    expected_columns = ["author", "review", "score", "date", "app_version"]
    for column in expected_columns:
        if column not in normalized_df.columns:
            normalized_df[column] = ""
    normalized_df["review"] = normalized_df["review"].fillna("").astype(str)
    normalized_df["score"] = pd.to_numeric(normalized_df["score"], errors="coerce")
    normalized_df["date"] = safe_to_datetime(normalized_df["date"])
    return normalized_df


def top_keywords_from_reviews(series: pd.Series, top_n: int = 10, use_cleaning: bool = True) -> List[tuple]:
    all_words = []
    for text in series.fillna("").astype(str).tolist():
        all_words.extend(tokenize_review_text(text, use_cleaning=use_cleaning))
    return Counter(all_words).most_common(top_n)


def make_wordcloud(text: str, is_dark_mode: bool = False, use_cleaning: bool = True) -> Optional[plt.Figure]:
    prepared_text = clean_review_text_for_analysis(text) if use_cleaning else text
    if not prepared_text.strip():
        return None
    background_color = "#0b1220" if is_dark_mode else "#ffffff"
    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color=background_color,
        collocations=False,
        max_words=120,
        font_path=resolve_wordcloud_font(),
    ).generate(prepared_text)
    figure = plt.figure(figsize=(12, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return figure


def build_score_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    counts = (
        dataframe["score"].dropna().astype(int).value_counts()
        .reindex([1, 2, 3, 4, 5])
        .fillna(0)
        .astype(int)
    )
    return pd.DataFrame({"count": counts.values}, index=[1, 2, 3, 4, 5])


def build_sentiment_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    sentiment_dist = (
        dataframe["score"].apply(score_to_label).dropna().value_counts()
        .reindex(["positive", "neutral", "negative"])
        .fillna(0)
        .astype(int)
    )
    return sentiment_dist.rename_axis("감성").reset_index(name="개수")


def build_period_keyword_table(dataframe: pd.DataFrame, granularity: str, use_cleaning: bool = True) -> pd.DataFrame:
    df_work = dataframe.dropna(subset=["date"]).copy()
    if granularity.startswith("일"):
        df_work["_period"] = df_work["date"].dt.date.astype(str)
    elif granularity.startswith("주"):
        df_work["_period"] = df_work["date"].dt.to_period("W").astype(str)
    else:
        df_work["_period"] = df_work["date"].dt.to_period("M").astype(str)
    period_rows = []
    for period_value, group in df_work.groupby("_period"):
        keywords = top_keywords_from_reviews(group["review"], top_n=3, use_cleaning=use_cleaning)
        period_rows.append(
            {
                "period": period_value,
                "reviews": len(group),
                "avg_score": round(float(group["score"].mean()), 2) if group["score"].notna().any() else None,
                "top_keywords": ", ".join([word for word, _ in keywords]) if keywords else "-",
            }
        )
    return pd.DataFrame(period_rows).sort_values("period")


def build_daily_trend(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.dropna(subset=["date"])
        .assign(일자=lambda data: data["date"].dt.date)
        .groupby("일자")
        .size()
        .reset_index(name="리뷰수")
        .sort_values("일자")
        .set_index("일자")
    )
