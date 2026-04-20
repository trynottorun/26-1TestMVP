from collections import Counter

import pandas as pd

from review_core.analysis import keyword_counts_from_reviews, normalize_analysis_dataframe, top_keywords_from_reviews
from review_core.shared import score_to_label


def compare_scores(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> pd.DataFrame:
    data1 = normalize_analysis_dataframe(df1)
    data2 = normalize_analysis_dataframe(df2)
    return pd.DataFrame(
        {
            "file": [label1, label2],
            "avg_score": [
                round(float(data1["score"].mean()), 2) if data1["score"].notna().any() else None,
                round(float(data2["score"].mean()), 2) if data2["score"].notna().any() else None,
            ],
            "review_count": [len(data1), len(data2)],
        }
    )


def compare_keywords(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    top_n: int = 10,
    use_cleaning: bool = True,
) -> pd.DataFrame:
    data1 = normalize_analysis_dataframe(df1)
    data2 = normalize_analysis_dataframe(df2)
    return pd.DataFrame(
        {
            label1: [", ".join([word for word, _ in top_keywords_from_reviews(data1["review"], top_n=top_n, use_cleaning=use_cleaning)]) or "-"],
            label2: [", ".join([word for word, _ in top_keywords_from_reviews(data2["review"], top_n=top_n, use_cleaning=use_cleaning)]) or "-"],
        }
    )


def compare_sentiment(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> pd.DataFrame:
    rows = []
    for label, raw_df in [(label1, df1), (label2, df2)]:
        dataset = normalize_analysis_dataframe(raw_df)
        sentiment_dist = dataset["score"].apply(score_to_label).dropna().value_counts()
        total = max(int(sentiment_dist.sum()), 1)
        rows.append(
            {
                "file": label,
                "positive": round(int(sentiment_dist.get("positive", 0)) / total * 100, 1),
                "neutral": round(int(sentiment_dist.get("neutral", 0)) / total * 100, 1),
                "negative": round(int(sentiment_dist.get("negative", 0)) / total * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def compare_trends(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, granularity: str) -> pd.DataFrame:
    data1 = normalize_analysis_dataframe(df1)
    data2 = normalize_analysis_dataframe(df2)

    def build_trend(dataset: pd.DataFrame, label: str) -> pd.DataFrame:
        valid_dataset = dataset.dropna(subset=["date"]).copy()
        granularity_lower = str(granularity).lower()
        if granularity_lower.startswith("day") or "day" in granularity_lower or "일" in str(granularity):
            valid_dataset["period"] = valid_dataset["date"].dt.date.astype(str)
        elif granularity_lower.startswith("week") or "week" in granularity_lower or "주" in str(granularity):
            valid_dataset["period"] = valid_dataset["date"].dt.to_period("W").astype(str)
        else:
            valid_dataset["period"] = valid_dataset["date"].dt.to_period("M").astype(str)
        return valid_dataset.groupby("period").size().reset_index(name=label)

    trend1 = build_trend(data1, label1)
    trend2 = build_trend(data2, label2)
    return pd.merge(trend1, trend2, on="period", how="outer").fillna(0).sort_values("period")


def build_compare_summary(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, use_cleaning: bool = True) -> list[str]:
    data1 = normalize_analysis_dataframe(df1)
    data2 = normalize_analysis_dataframe(df2)
    avg1 = float(data1["score"].mean()) if data1["score"].notna().any() else 0.0
    avg2 = float(data2["score"].mean()) if data2["score"].notna().any() else 0.0
    neg1 = int(data1["score"].apply(score_to_label).eq("negative").sum())
    neg2 = int(data2["score"].apply(score_to_label).eq("negative").sum())
    top1 = [word for word, _ in top_keywords_from_reviews(data1["review"], top_n=3, use_cleaning=use_cleaning)]
    top2 = [word for word, _ in top_keywords_from_reviews(data2["review"], top_n=3, use_cleaning=use_cleaning)]
    return [
        f"평균 평점은 `{label1}` {avg1:.2f}점, `{label2}` {avg2:.2f}점입니다.",
        f"부정 리뷰 수는 `{label1}` {neg1}건, `{label2}` {neg2}건입니다.",
        f"`{label1}` 핵심 키워드: {', '.join(top1) if top1 else '-'}",
        f"`{label2}` 핵심 키워드: {', '.join(top2) if top2 else '-'}",
    ]


def compare_common_and_distinct_keywords(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    top_n: int = 10,
    use_cleaning: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data1 = normalize_analysis_dataframe(df1)
    data2 = normalize_analysis_dataframe(df2)

    counter1: Counter = keyword_counts_from_reviews(data1["review"], use_cleaning=use_cleaning)
    counter2: Counter = keyword_counts_from_reviews(data2["review"], use_cleaning=use_cleaning)

    common_keywords = []
    for keyword in set(counter1.keys()) & set(counter2.keys()):
        common_keywords.append(
            {
                "keyword": keyword,
                label1: int(counter1[keyword]),
                label2: int(counter2[keyword]),
                "total": int(counter1[keyword] + counter2[keyword]),
            }
        )
    common_df = pd.DataFrame(common_keywords).sort_values("total", ascending=False).head(top_n) if common_keywords else pd.DataFrame(columns=["keyword", label1, label2, "total"])

    distinct_rows = []
    all_keywords = set(counter1.keys()) | set(counter2.keys())
    for keyword in all_keywords:
        count1 = int(counter1.get(keyword, 0))
        count2 = int(counter2.get(keyword, 0))
        if count1 == 0 and count2 == 0:
            continue
        distinct_rows.append(
            {
                "keyword": keyword,
                label1: count1,
                label2: count2,
                "gap": abs(count1 - count2),
                "dominant_game": label1 if count1 > count2 else label2 if count2 > count1 else "same",
            }
        )
    distinct_df = pd.DataFrame(distinct_rows).sort_values("gap", ascending=False).head(top_n) if distinct_rows else pd.DataFrame(columns=["keyword", label1, label2, "gap", "dominant_game"])
    return common_df, distinct_df
