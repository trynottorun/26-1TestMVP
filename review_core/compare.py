import pandas as pd

from review_core.analysis import build_period_keyword_table, normalize_analysis_dataframe, top_keywords_from_reviews
from review_core.shared import score_to_label


def compare_scores(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> pd.DataFrame:
    data1 = normalize_analysis_dataframe(df1)
    data2 = normalize_analysis_dataframe(df2)
    return pd.DataFrame(
        {
            "파일": [label1, label2],
            "평균 평점": [
                round(float(data1["score"].mean()), 2) if data1["score"].notna().any() else None,
                round(float(data2["score"].mean()), 2) if data2["score"].notna().any() else None,
            ],
            "리뷰 수": [len(data1), len(data2)],
        }
    )


def compare_keywords(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, top_n: int = 10, use_cleaning: bool = True) -> pd.DataFrame:
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
                "파일": label,
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
        if granularity.startswith("일"):
            valid_dataset["period"] = valid_dataset["date"].dt.date.astype(str)
        elif granularity.startswith("주"):
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
        f"평점은 `{label1}`가 {avg1:.2f}점, `{label2}`가 {avg2:.2f}점입니다.",
        f"부정 리뷰 수는 `{label1}` {neg1}건, `{label2}` {neg2}건입니다.",
        f"`{label1}` 핵심 키워드: {', '.join(top1) if top1 else '-'}",
        f"`{label2}` 핵심 키워드: {', '.join(top2) if top2 else '-'}",
    ]
