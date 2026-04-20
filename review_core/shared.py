import os
import re
from typing import List, Optional
from urllib.request import ProxyHandler, build_opener, install_opener

import pandas as pd


PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]

ANALYSIS_STOPWORDS = {
    "the", "and", "to", "of", "is", "in", "it", "this", "that", "for",
    "with", "you", "your", "from", "have", "has", "are", "was", "were",
    "있어요", "좋아요", "입니다", "그리고", "하는", "에서", "으로", "에게",
    "너무", "진짜", "정말", "완전", "엄청", "매우", "그냥", "약간", "조금",
    "좀", "많이", "살짝", "와", "헉", "아", "음", "오", "근데", "하지만",
    "그런데", "같아요", "했어요", "입니다", "네요", "해요", "했는데", "있고",
    "있음", "없음", "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ",
}

KEYWORD_SYNONYM_MAP = {
    "강종": "튕김",
    "꺼짐": "튕김",
    "팅김": "튕김",
    "튕겨요": "튕김",
    "버벅임": "렉",
    "끊김": "렉",
    "끊겨요": "렉",
    "버벅거림": "렉",
    "현질": "과금",
    "과금유도": "과금",
    "pay": "과금",
    "매치": "매칭",
    "매칭안됨": "매칭",
    "접속불가": "접속",
    "로그인": "접속",
    "로그인안됨": "접속",
    "서버렉": "서버",
}


def disable_broken_proxy_settings() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    install_opener(build_opener(ProxyHandler({})))


def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    cleaned = text.replace("\n", " ").replace("\r", " ").strip().lower()
    for character in re.findall(r"[^\w가-힣]", cleaned):
        cleaned = cleaned.replace(character, " ")
    return [word for word in cleaned.split() if len(word) >= 2]


def clean_review_text_for_analysis(text: str) -> str:
    if not isinstance(text, str):
        return ""

    cleaned = text.strip().lower()
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"[ㅋㅎㅠㅜ]{2,}", " ", cleaned)
    cleaned = re.sub(r"[^\w가-힣\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_review_text(text: str, use_cleaning: bool = True) -> List[str]:
    base_text = clean_review_text_for_analysis(text) if use_cleaning else text
    tokens = simple_tokenize(base_text)
    if not use_cleaning:
        return tokens
    normalized_tokens = [KEYWORD_SYNONYM_MAP.get(token, token) for token in tokens]
    return [token for token in normalized_tokens if token not in ANALYSIS_STOPWORDS and len(token) >= 2]


def score_to_label(value: float) -> Optional[str]:
    try:
        score_value = float(value)
    except Exception:
        return None
    if score_value >= 4:
        return "positive"
    if score_value == 3:
        return "neutral"
    return "negative"
