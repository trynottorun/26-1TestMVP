from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st
from google_play_scraper import Sort, reviews, search


st.set_page_config(
    page_title="Google Play Review Collector",
    page_icon="P",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
        }
        .block-container {
            max-width: 860px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .hero-card, .result-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-bottom: 0;
            line-height: 1.6;
        }
        .section-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .selected-app-card {
            margin-top: 0.8rem;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .selected-app-card p {
            margin: 0.15rem 0;
            color: #0f172a;
        }
        .summary-text {
            color: #0f172a;
            font-weight: 600;
            margin: 0;
        }
        .hint-text {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.8rem;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 14px;
            min-height: 2.8rem;
            font-weight: 600;
        }
        @media (max-width: 640px) {
            .block-container {
                padding-top: 1.2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero-card, .result-card {
                padding: 1rem;
                border-radius: 18px;
            }
            .hero-title {
                font-size: 1.6rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults = {
        "search_keyword": "",
        "search_results": [],
        "selected_candidate_index": 0,
        "selected_app_name": "",
        "selected_app_id": "",
        "reviews_df": None,
        "last_loaded_count": 0,
        "show_table": False,
        "search_error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_review_state() -> None:
    st.session_state.reviews_df = None
    st.session_state.last_loaded_count = 0
    st.session_state.show_table = False


def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def search_google_play_apps(keyword: str, limit: int) -> List[Dict[str, str]]:
    search_results = search(
        keyword,
        n_hits=limit,
        lang="ko",
        country="kr",
    )

    candidates = []
    for item in search_results[:limit]:
        app_name = item.get("title", "").strip()
        app_id = item.get("appId", "").strip()
        if app_name and app_id:
            candidates.append({"title": app_name, "appId": app_id})

    return candidates


def fetch_google_play_reviews(app_id: str, review_count: int) -> pd.DataFrame:
    fetched_reviews, _ = reviews(
        app_id,
        lang="ko",
        country="kr",
        sort=Sort.NEWEST,
        count=review_count,
    )

    records = []
    for item in fetched_reviews:
        review_date = item.get("at")
        formatted_date = (
            review_date.strftime("%Y-%m-%d %H:%M")
            if isinstance(review_date, datetime)
            else ""
        )
        records.append(
            {
                "Author": item.get("userName", ""),
                "Review": item.get("content", ""),
                "Rating": item.get("score", ""),
                "Created At": formatted_date,
                "App Version": item.get("reviewCreatedVersion") or "Not available",
            }
        )

    return pd.DataFrame(records)


def handle_app_search() -> None:
    keyword = st.session_state.search_keyword.strip()
    st.session_state.search_error = ""
    st.session_state.search_results = []
    st.session_state.selected_candidate_index = 0
    st.session_state.selected_app_name = ""
    st.session_state.selected_app_id = ""
    reset_review_state()

    if not keyword:
        st.session_state.search_error = "Please enter an app name to search."
        return

    try:
        candidates = search_google_play_apps(keyword, 5)
    except Exception:
        st.session_state.search_error = "Something went wrong while searching for apps. Please try again."
        return

    if not candidates:
        st.session_state.search_error = "No matching apps were found. Try a different keyword."
        return

    st.session_state.search_results = candidates
    st.session_state.selected_app_name = candidates[0]["title"]
    st.session_state.selected_app_id = candidates[0]["appId"]


def handle_candidate_change() -> None:
    candidates = st.session_state.search_results
    selected_index = int(st.session_state.selected_candidate_index)

    if 0 <= selected_index < len(candidates):
        st.session_state.selected_app_name = candidates[selected_index]["title"]
        st.session_state.selected_app_id = candidates[selected_index]["appId"]
        reset_review_state()


init_session_state()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Google Play Review Collector</div>
        <p class="hero-subtitle">
            Search by app name, choose the closest match from the top results,
            and collect recent Google Play reviews in just a few clicks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.container():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">App Search</div>', unsafe_allow_html=True)

    search_col, button_col = st.columns([4, 1.2])
    with search_col:
        st.text_input(
            "App name",
            key="search_keyword",
            placeholder="Example: KakaoTalk, Naver, Instagram",
        )
    with button_col:
        st.button(
            "Find Apps",
            type="primary",
            use_container_width=True,
            on_click=handle_app_search,
        )

    if st.session_state.search_error:
        st.warning(st.session_state.search_error)

    if st.session_state.search_results:
        options = list(range(len(st.session_state.search_results)))

        def format_option(index: int) -> str:
            item = st.session_state.search_results[index]
            return "{0} ({1})".format(item["title"], item["appId"])

        st.radio(
            "Top 5 matching apps",
            options=options,
            key="selected_candidate_index",
            format_func=format_option,
            on_change=handle_candidate_change,
        )

        st.markdown(
            """
            <div class="selected-app-card">
                <p><strong>Selected app</strong></p>
                <p>{0}</p>
                <p>App ID: {1}</p>
            </div>
            """.format(
                st.session_state.selected_app_name,
                st.session_state.selected_app_id,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

with st.container():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Review Collection</div>', unsafe_allow_html=True)

    review_count = st.number_input(
        "Number of reviews to collect",
        min_value=1,
        max_value=1000,
        value=50,
        step=10,
        help="You can collect up to 1,000 reviews at a time.",
    )

    load_reviews = st.button("Collect Reviews", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


if load_reviews:
    selected_app_id = st.session_state.selected_app_id.strip()

    if not selected_app_id:
        st.warning("Please search for an app and select one of the results first.")
        reset_review_state()
    else:
        with st.spinner("Collecting reviews for the selected app. Please wait a moment."):
            try:
                reviews_df = fetch_google_play_reviews(selected_app_id, int(review_count))

                if reviews_df.empty:
                    st.info("No reviews were found for the selected app.")
                    reset_review_state()
                else:
                    st.session_state.reviews_df = reviews_df
                    st.session_state.last_loaded_count = len(reviews_df)
                    st.success("Reviews were collected successfully.")
            except Exception:
                reset_review_state()
                st.error("An error occurred while collecting reviews. Please try again.")


if st.session_state.reviews_df is not None and not st.session_state.reviews_df.empty:
    st.write("")
    st.markdown(
        """
        <div class="result-card">
            <p class="summary-text">Collected {0} reviews.</p>
            <p class="hint-text">{1} ({2})</p>
        </div>
        """.format(
            st.session_state.last_loaded_count,
            st.session_state.selected_app_name,
            st.session_state.selected_app_id,
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    preview_col, download_col = st.columns(2)

    with preview_col:
        if st.button("Preview Reviews", use_container_width=True):
            st.session_state.show_table = not st.session_state.show_table

    with download_col:
        st.download_button(
            label="Download CSV",
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
