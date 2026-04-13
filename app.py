from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from google_play_scraper import Sort, reviews, search


st.set_page_config(
    page_title="Google play store 리뷰 수집기",
    page_icon="📝",
    layout="centered",
)


PERIOD_OPTIONS = [
    ("all", "전체 기간"),
    ("1m", "최근 1개월"),
    ("2m", "최근 2개월"),
    ("3m", "최근 3개월"),
]


def init_session_state() -> None:
    defaults = {
        "search_keyword": "",
        "search_results": [],
        "selected_candidate_index": 0,
        "selected_app_name": "",
        "selected_package_id": "",
        "review_count": 50,
        "selected_period": "all",
        "reviews_df": None,
        "last_loaded_count": 0,
        "show_table": False,
        "search_error": "",
        "fetch_error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, #f3f7ff 0%, rgba(243, 247, 255, 0) 32%),
                    linear-gradient(180deg, #f6f7fb 0%, #eef2f7 100%);
            }
            .block-container {
                max-width: 880px;
                padding-top: 2.5rem;
                padding-bottom: 3rem;
            }
            .card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 26px;
                padding: 1.5rem;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
                backdrop-filter: blur(12px);
            }
            .hero-title {
                font-size: 2.2rem;
                font-weight: 700;
                color: #111827;
                margin: 0 0 0.6rem 0;
                letter-spacing: -0.02em;
            }
            .hero-copy {
                margin: 0;
                color: #4b5563;
                line-height: 1.7;
                font-size: 1rem;
            }
            .section-title {
                margin: 0 0 0.9rem 0;
                color: #111827;
                font-size: 1.05rem;
                font-weight: 650;
            }
            .label-text {
                display: block;
                margin-bottom: 0.45rem;
                color: #374151;
                font-size: 0.95rem;
                font-weight: 600;
            }
            .selection-card {
                margin-top: 1rem;
                padding: 0.95rem 1rem;
                border-radius: 18px;
                background: #f8fafc;
                border: 1px solid #e5e7eb;
            }
            .selection-card p {
                margin: 0.1rem 0;
                color: #111827;
            }
            .meta-text {
                color: #6b7280;
                font-size: 0.93rem;
            }
            .period-label {
                margin: 0.4rem 0 0.7rem 0;
                color: #374151;
                font-size: 0.95rem;
                font-weight: 600;
            }
            .summary-card {
                margin-top: 1rem;
                padding: 1rem 1.1rem;
                border-radius: 20px;
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e5e7eb;
            }
            .summary-card p {
                margin: 0.2rem 0;
            }
            div[data-testid="stHorizontalBlock"] {
                gap: 0.75rem;
            }
            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button {
                border-radius: 14px;
                border: 1px solid #d1d5db;
                min-height: 2.8rem;
                font-weight: 600;
                background: #ffffff;
            }
            div[data-testid="stButton"] > button[kind="primary"] {
                background: #111827;
                color: #ffffff;
                border: none;
            }
            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input {
                border-radius: 14px;
                background: #ffffff;
            }
            div[role="radiogroup"] label {
                padding: 0.4rem 0;
            }
            @media (max-width: 640px) {
                .block-container {
                    padding-top: 1.25rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .card {
                    padding: 1.15rem;
                    border-radius: 20px;
                }
                .hero-title {
                    font-size: 1.7rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_review_state() -> None:
    st.session_state.reviews_df = None
    st.session_state.last_loaded_count = 0
    st.session_state.show_table = False
    st.session_state.fetch_error = ""


def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def get_cutoff_datetime(period_key: str) -> Optional[datetime]:
    now = datetime.now()

    if period_key == "1m":
        return now - timedelta(days=30)
    if period_key == "2m":
        return now - timedelta(days=60)
    if period_key == "3m":
        return now - timedelta(days=90)
    return None


def search_apps(keyword: str, limit: int) -> List[Dict[str, str]]:
    results = search(
        keyword,
        n_hits=limit,
        lang="ko",
        country="kr",
    )

    candidates = []
    for item in results[:limit]:
        app_title = item.get("title", "").strip()
        package_id = item.get("appId", "").strip()
        if not app_title or not package_id:
            continue
        candidates.append(
            {
                "title": app_title,
                "appId": package_id,
            }
        )
    return candidates


def fetch_google_play_reviews(
    app_id: str,
    review_count: int,
    period_key: str,
) -> pd.DataFrame:
    cutoff_datetime = get_cutoff_datetime(period_key)
    continuation_token = None
    filtered_reviews = []
    page_size = min(max(review_count, 100), 200)
    max_requests = 10

    for _ in range(max_requests):
        fetched_reviews, continuation_token = reviews(
            app_id,
            lang="ko",
            country="kr",
            sort=Sort.NEWEST,
            count=page_size,
            continuation_token=continuation_token,
        )

        if not fetched_reviews:
            break

        stop_paging = False

        for item in fetched_reviews:
            review_date = item.get("at")
            if cutoff_datetime and isinstance(review_date, datetime) and review_date < cutoff_datetime:
                stop_paging = True
                continue

            formatted_date = (
                review_date.strftime("%Y-%m-%d %H:%M")
                if isinstance(review_date, datetime)
                else ""
            )
            filtered_reviews.append(
                {
                    "작성자명": item.get("userName", ""),
                    "리뷰 내용": item.get("content", ""),
                    "평점": item.get("score", ""),
                    "작성일": formatted_date,
                    "앱 버전": item.get("reviewCreatedVersion") or "정보 없음",
                }
            )

            if len(filtered_reviews) >= review_count:
                break

        if len(filtered_reviews) >= review_count or stop_paging or continuation_token is None:
            break

    return pd.DataFrame(filtered_reviews[:review_count])


def handle_app_search() -> None:
    keyword = st.session_state.search_keyword.strip()
    reset_review_state()
    st.session_state.search_results = []
    st.session_state.selected_candidate_index = 0
    st.session_state.selected_app_name = ""
    st.session_state.selected_package_id = ""
    st.session_state.search_error = ""

    if not keyword:
        st.session_state.search_error = "앱 이름을 먼저 입력해 주세요."
        return

    try:
        candidates = search_apps(keyword, 5)
    except Exception:
        st.session_state.search_error = "앱 검색 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        return

    if not candidates:
        st.session_state.search_error = "검색된 앱이 없습니다. 다른 이름으로 다시 찾아보세요."
        return

    st.session_state.search_results = candidates
    st.session_state.selected_app_name = candidates[0]["title"]
    st.session_state.selected_package_id = candidates[0]["appId"]


def handle_candidate_change() -> None:
    candidates = st.session_state.search_results
    selected_index = int(st.session_state.selected_candidate_index)

    if 0 <= selected_index < len(candidates):
        st.session_state.selected_app_name = candidates[selected_index]["title"]
        st.session_state.selected_package_id = candidates[selected_index]["appId"]
        reset_review_state()


def handle_period_change(period_key: str) -> None:
    st.session_state.selected_period = period_key
    reset_review_state()


def handle_fetch_reviews() -> None:
    package_id = st.session_state.selected_package_id.strip()
    review_count = int(st.session_state.review_count)
    period_key = st.session_state.selected_period
    reset_review_state()

    if not package_id:
        st.session_state.fetch_error = "먼저 앱을 검색하고 후보를 선택해 주세요."
        return

    try:
        reviews_df = fetch_google_play_reviews(package_id, review_count, period_key)
    except Exception:
        st.session_state.fetch_error = "리뷰를 수집하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        return

    if reviews_df.empty:
        st.session_state.fetch_error = "선택한 기간에 해당하는 리뷰를 찾지 못했습니다."
        return

    st.session_state.reviews_df = reviews_df
    st.session_state.last_loaded_count = len(reviews_df)


init_session_state()
render_styles()


st.markdown(
    """
    <div class="card">
        <h1 class="hero-title">Google play store 리뷰 수집기</h1>
        <p class="hero-copy">
            원하는 앱을 이름으로 찾고, 기간별로 필요한 리뷰만 빠르게 수집할 수 있는 도구입니다.
            수집한 리뷰는 서비스 개선, VOC 분석, 경쟁 앱 모니터링 같은 작업에 바로 활용할 수 있습니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">1. 앱 찾기</p>', unsafe_allow_html=True)
    st.markdown('<span class="label-text">앱 이름</span>', unsafe_allow_html=True)

    search_col, button_col = st.columns([4, 1.2])
    with search_col:
        st.text_input(
            "앱 이름 입력",
            key="search_keyword",
            placeholder="예: 카카오톡, 네이버, 인스타그램",
            label_visibility="collapsed",
        )
    with button_col:
        st.button(
            "앱 찾기",
            type="primary",
            use_container_width=True,
            on_click=handle_app_search,
        )

    if st.session_state.search_error:
        st.warning(st.session_state.search_error)

    if st.session_state.search_results:
        radio_options = list(range(len(st.session_state.search_results)))

        def option_label(index: int) -> str:
            item = st.session_state.search_results[index]
            return "{0} ({1})".format(item["title"], item["appId"])

        st.radio(
            "검색 결과",
            options=radio_options,
            key="selected_candidate_index",
            format_func=option_label,
            on_change=handle_candidate_change,
        )

        st.markdown(
            """
            <div class="selection-card">
                <p><strong>선택된 앱</strong></p>
                <p>{0}</p>
                <p class="meta-text">패키지명: {1}</p>
            </div>
            """.format(
                st.session_state.selected_app_name,
                st.session_state.selected_package_id,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">2. 리뷰 수집 조건</p>', unsafe_allow_html=True)

    st.number_input(
        "리뷰 개수",
        key="review_count",
        min_value=1,
        max_value=1000,
        step=10,
        help="최대 1,000개까지 수집할 수 있습니다.",
    )

    st.markdown('<p class="period-label">기간 선택</p>', unsafe_allow_html=True)
    period_columns = st.columns(len(PERIOD_OPTIONS))

    for index, option in enumerate(PERIOD_OPTIONS):
        period_key, period_label = option
        is_selected = st.session_state.selected_period == period_key
        button_type = "primary" if is_selected else "secondary"

        with period_columns[index]:
            st.button(
                period_label,
                key="period_{0}".format(period_key),
                type=button_type,
                use_container_width=True,
                on_click=handle_period_change,
                args=(period_key,),
            )

    st.button(
        "리뷰 수집",
        type="primary",
        use_container_width=True,
        on_click=handle_fetch_reviews,
    )

    if st.session_state.fetch_error:
        st.warning(st.session_state.fetch_error)

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.reviews_df is not None and not st.session_state.reviews_df.empty:
    st.write("")
    st.markdown(
        """
        <div class="card summary-card">
            <p><strong>{0}개 리뷰를 수집했습니다.</strong></p>
            <p class="meta-text">선택 앱: {1}</p>
            <p class="meta-text">패키지명: {2}</p>
        </div>
        """.format(
            st.session_state.last_loaded_count,
            st.session_state.selected_app_name,
            st.session_state.selected_package_id,
        ),
        unsafe_allow_html=True,
    )

    preview_col, download_col = st.columns(2)

    with preview_col:
        if st.button("미리보기", use_container_width=True):
            st.session_state.show_table = not st.session_state.show_table

    with download_col:
        st.download_button(
            label="CSV 다운로드",
            data=convert_df_to_csv(st.session_state.reviews_df),
            file_name="google_play_reviews.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.show_table:
        st.dataframe(
            st.session_state.reviews_df,
            use_container_width=True,
            hide_index=True,
        )
