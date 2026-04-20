from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from google_play_scraper import Sort, app, reviews


COUNTRY_OPTIONS = {
    "한국": ("ko", "kr"),
    "미국": ("en", "us"),
    "일본": ("ja", "jp"),
    "대만": ("zh_TW", "tw"),
}

PLAY_STORE_SEARCH_URL = "https://play.google.com/store/search?q={query}&c=apps&hl={lang}&gl={country}"
APP_ID_PATTERN = re.compile(r"/store/apps/details\?id=([^&\"\\\\]+)")


def normalize_search_text(value: str) -> str:
    return "".join(str(value or "").lower().split())


def get_search_priority(keyword: str, title: str, app_id: str) -> tuple:
    normalized_keyword = normalize_search_text(keyword)
    normalized_title = normalize_search_text(title)
    normalized_app_id = normalize_search_text(app_id)
    return (
        0 if normalized_title == normalized_keyword else 1,
        0 if normalized_title.startswith(normalized_keyword) else 1,
        0 if normalized_keyword in normalized_title else 1,
        0 if normalized_app_id == normalized_keyword else 1,
        0 if normalized_keyword in normalized_app_id else 1,
        abs(len(normalized_title) - len(normalized_keyword)),
        normalized_title,
    )


def get_search_locales(country_label: str) -> List[tuple]:
    primary_locale = COUNTRY_OPTIONS.get(country_label, COUNTRY_OPTIONS["한국"])
    fallback_locales = [locale for locale in COUNTRY_OPTIONS.values() if locale != primary_locale]
    return [primary_locale] + fallback_locales


def fetch_search_page_app_ids(keyword: str, lang: str, country: str) -> List[str]:
    search_url = PLAY_STORE_SEARCH_URL.format(query=quote(keyword), lang=lang, country=country)
    request = Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
    response_html = urlopen(request, timeout=20).read().decode("utf-8", errors="ignore")
    app_ids = []
    seen_app_ids = set()
    for app_id_value in APP_ID_PATTERN.findall(response_html):
        normalized_app_id = str(app_id_value or "").strip()
        if not normalized_app_id or normalized_app_id in seen_app_ids:
            continue
        seen_app_ids.add(normalized_app_id)
        app_ids.append(normalized_app_id)
    return app_ids


def get_app_metadata(app_id_value: str, country_label: str = "한국") -> Dict[str, str]:
    normalized_app_id = app_id_value.strip()
    if not normalized_app_id:
        return {}
    last_url_error = None
    for lang, country in get_search_locales(country_label):
        try:
            result = app(normalized_app_id, lang=lang, country=country)
        except URLError as error:
            last_url_error = error
            continue
        except Exception:
            continue
        app_title = str(result.get("title") or normalized_app_id).strip()
        package_id = str(result.get("appId") or normalized_app_id).strip()
        if app_title and package_id:
            return {"title": app_title, "appId": package_id}
    if last_url_error is not None:
        raise last_url_error
    return {}


def search_apps(keyword: str, country_label: str = "한국", limit: int = 5) -> List[Dict[str, str]]:
    normalized_keyword = keyword.strip()
    if not normalized_keyword:
        return []
    candidates = []
    seen_app_ids = set()
    last_url_error = None
    for lang, country in get_search_locales(country_label):
        try:
            candidate_app_ids = fetch_search_page_app_ids(normalized_keyword, lang, country)
        except URLError as error:
            last_url_error = error
            continue
        for package_id in candidate_app_ids:
            if package_id in seen_app_ids:
                continue
            app_info = get_app_metadata(package_id, country_label=country_label)
            app_title = str(app_info.get("title") or "").strip()
            verified_app_id = str(app_info.get("appId") or package_id).strip()
            if not app_title or not verified_app_id or verified_app_id in seen_app_ids:
                continue
            seen_app_ids.add(verified_app_id)
            candidates.append({"title": app_title, "appId": verified_app_id})
    if not candidates and last_url_error is not None:
        raise last_url_error
    return sorted(
        candidates,
        key=lambda item: get_search_priority(normalized_keyword, item["title"], item["appId"]),
    )[:limit]


def get_cutoff_datetime(period_key: str) -> Optional[datetime]:
    now = datetime.now()
    if period_key == "1m":
        return now - timedelta(days=30)
    if period_key == "2m":
        return now - timedelta(days=60)
    if period_key == "3m":
        return now - timedelta(days=90)
    return None


def fetch_google_play_reviews(app_id_value: str, review_count: int, period_key: str = "all") -> pd.DataFrame:
    cutoff_datetime = get_cutoff_datetime(period_key)
    continuation_token = None
    collected_rows = []
    page_size = min(max(review_count, 100), 200)
    max_requests = max(10, (review_count // page_size) + 5)
    for _ in range(max_requests):
        fetched_reviews, continuation_token = reviews(
            app_id_value,
            lang="ko",
            country="kr",
            sort=Sort.NEWEST,
            count=page_size,
            continuation_token=continuation_token,
        )
        if not fetched_reviews:
            break
        reached_older_reviews = False
        for item in fetched_reviews:
            review_date = item.get("at")
            review_content = (item.get("content") or "").strip()
            if cutoff_datetime and isinstance(review_date, datetime) and review_date < cutoff_datetime:
                reached_older_reviews = True
                continue
            if not review_content:
                continue
            formatted_date = review_date.strftime("%Y-%m-%d %H:%M") if isinstance(review_date, datetime) else ""
            collected_rows.append(
                {
                    "작성자명": item.get("userName", ""),
                    "리뷰 내용": review_content,
                    "평점": item.get("score", ""),
                    "작성일": formatted_date,
                    "앱 버전": item.get("reviewCreatedVersion") or "정보 없음",
                }
            )
            if len(collected_rows) >= review_count:
                break
        if len(collected_rows) >= review_count or reached_older_reviews or continuation_token is None:
            break
    return pd.DataFrame(collected_rows[:review_count])
