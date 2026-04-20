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
        "????": "author",
        "?? ??": "review",
        "??": "score",
        "???": "date",
        "???:": "date",
        "? ??": "app_version",
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
    counter = keyword_counts_from_reviews(series, use_cleaning=use_cleaning)
    return counter.most_common(top_n)


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
    return sentiment_dist.rename_axis("sentiment").reset_index(name="count")


def _build_period_value(date_series: pd.Series, granularity: str) -> pd.Series:
    granularity_lower = str(granularity).lower()
    if granularity_lower.startswith("day") or "day" in granularity_lower or "일" in str(granularity):
        return date_series.dt.date.astype(str)
    if granularity_lower.startswith("week") or "week" in granularity_lower or "주" in str(granularity):
        return date_series.dt.to_period("W").astype(str)
    return date_series.dt.to_period("M").astype(str)


def build_period_keyword_table(dataframe: pd.DataFrame, granularity: str, use_cleaning: bool = True) -> pd.DataFrame:
    df_work = normalize_analysis_dataframe(dataframe)
    if df_work is None or df_work.empty:
        return pd.DataFrame(columns=["period", "reviews", "avg_score", "top_keywords"])

    df_work = df_work.dropna(subset=["date"]).copy()
    if df_work.empty:
        return pd.DataFrame(columns=["period", "reviews", "avg_score", "top_keywords"])

    df_work["_period"] = _build_period_value(df_work["date"], granularity)
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
    df_work = normalize_analysis_dataframe(dataframe)
    if df_work is None or df_work.empty:
        return pd.DataFrame(columns=["review_count"])
    return (
        df_work.dropna(subset=["date"])
        .assign(date_value=lambda data: data["date"].dt.date)
        .groupby("date_value")
        .size()
        .reset_index(name="review_count")
        .sort_values("date_value")
        .set_index("date_value")
    )


def keyword_counts_from_reviews(series: pd.Series, use_cleaning: bool = True) -> Counter:
    counter: Counter = Counter()
    for text in series.fillna("").astype(str).tolist():
        counter.update(tokenize_review_text(text, use_cleaning=use_cleaning))
    return counter


def build_keyword_frequency_over_time(
    dataframe: pd.DataFrame,
    selected_keywords: list[str],
    granularity: str = "Week",
    use_cleaning: bool = True,
) -> pd.DataFrame:
    df_work = normalize_analysis_dataframe(dataframe)
    if df_work is None or df_work.empty or not selected_keywords:
        return pd.DataFrame(columns=["period", "keyword", "count"])

    valid = df_work.dropna(subset=["date"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["period", "keyword", "count"])

    valid["period"] = _build_period_value(valid["date"], granularity)
    normalized_keywords = [keyword.strip().lower() for keyword in selected_keywords if keyword and keyword.strip()]

    rows: list[dict] = []
    for period_value, group in valid.groupby("period"):
        token_lists = [
            tokenize_review_text(review_text, use_cleaning=use_cleaning)
            for review_text in group["review"].fillna("").astype(str).tolist()
        ]
        for keyword in normalized_keywords:
            count = sum(tokens.count(keyword) for tokens in token_lists)
            rows.append({"period": period_value, "keyword": keyword, "count": int(count)})
    return pd.DataFrame(rows).sort_values(["period", "keyword"])


def filter_reviews_by_keyword(dataframe: pd.DataFrame, keyword: str, use_cleaning: bool = True) -> pd.DataFrame:
    df_work = normalize_analysis_dataframe(dataframe)
    if df_work is None or df_work.empty or not keyword:
        return pd.DataFrame()

    normalized_keyword = keyword.strip().lower()
    mask = df_work["review"].fillna("").astype(str).apply(
        lambda text: normalized_keyword in tokenize_review_text(text, use_cleaning=use_cleaning)
    )
    return df_work.loc[mask].copy()


def build_keyword_sentiment_distribution(dataframe: pd.DataFrame, keyword: str, use_cleaning: bool = True) -> pd.DataFrame:
    filtered = filter_reviews_by_keyword(dataframe, keyword, use_cleaning=use_cleaning)
    if filtered.empty:
        return pd.DataFrame({"sentiment": ["positive", "neutral", "negative"], "count": [0, 0, 0]})

    sentiment_dist = (
        filtered["score"].apply(score_to_label).dropna().value_counts()
        .reindex(["positive", "neutral", "negative"])
        .fillna(0)
        .astype(int)
    )
    return sentiment_dist.rename_axis("sentiment").reset_index(name="count")


def build_keyword_score_distribution(dataframe: pd.DataFrame, keyword: str, use_cleaning: bool = True) -> pd.DataFrame:
    filtered = filter_reviews_by_keyword(dataframe, keyword, use_cleaning=use_cleaning)
    if filtered.empty:
        return pd.DataFrame({"score": [1, 2, 3, 4, 5], "count": [0, 0, 0, 0, 0]})

    counts = (
        filtered["score"].dropna().astype(int).value_counts()
        .reindex([1, 2, 3, 4, 5])
        .fillna(0)
        .astype(int)
    )
    return pd.DataFrame({"score": [1, 2, 3, 4, 5], "count": counts.values})


def get_keyword_review_examples(
    dataframe: pd.DataFrame,
    keyword: str,
    use_cleaning: bool = True,
    limit: int = 5,
) -> pd.DataFrame:
    filtered = filter_reviews_by_keyword(dataframe, keyword, use_cleaning=use_cleaning)
    if filtered.empty:
        return pd.DataFrame(columns=["author", "review", "score", "date", "app_version"])

    columns = [column for column in ["author", "review", "score", "date", "app_version"] if column in filtered.columns]
    examples = filtered.sort_values("date", ascending=False, na_position="last").head(limit)[columns].copy()
    if "date" in examples.columns:
        examples["date"] = examples["date"].astype(str)
    return examples
