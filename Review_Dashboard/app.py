from pathlib import Path
import sys

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from review_core.analysis import (
    build_daily_trend,
    build_period_keyword_table,
    build_score_distribution,
    build_sentiment_distribution,
    make_wordcloud,
    normalize_analysis_dataframe,
    top_keywords_from_reviews,
)
from review_core.compare import (
    build_compare_summary,
    compare_keywords,
    compare_scores,
    compare_sentiment,
    compare_trends,
)
from review_core.crawler import COUNTRY_OPTIONS, fetch_google_play_reviews, get_app_metadata, search_apps
from review_core.shared import convert_df_to_csv, disable_broken_proxy_settings, score_to_label


st.set_page_config(page_title="Review Dashboard", page_icon="R", layout="wide")

CRAWLER_TAB = "crawler"
ANALYSIS_TAB = "analysis"
MAX_SELECTED_APPS = 2
MAX_REVIEW_COUNT = 5000
SLOT_IDS = [1, 2]


def init_session_state() -> None:
    defaults = {
        "current_tab": CRAWLER_TAB,
        "selected_country_label": "한국",
        "search_keyword": "",
        "search_results": [],
        "selected_candidate_index": 0,
        "selected_app_name": "",
        "app_id_input": "",
        "crawler_search_error": "",
        "crawler_fetch_error": "",
        "crawler_success_message": "",
        "selected_apps": [],
        "analysis_error": "",
        "analysis_compare_granularity": "주(Week)",
        "analysis_file_1_granularity": "주(Week)",
        "analysis_file_2_granularity": "주(Week)",
        "use_cleaning_compare": True,
        "use_cleaning_file_1": True,
        "use_cleaning_file_2": True,
    }
    for slot_id in SLOT_IDS:
        defaults[f"show_table_slot_{slot_id}"] = False
        defaults[f"review_count_slot_{slot_id}"] = 50
        defaults[f"reviews_df_slot_{slot_id}"] = None
        defaults[f"uploaded_df_slot_{slot_id}"] = None
        defaults[f"uploaded_name_slot_{slot_id}"] = ""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0) 26%),
                    linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
            }
            .block-container {
                max-width: 1180px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                gap: 10px;
            }
            .dashboard-card, .slot-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 22px;
                padding: 1.2rem;
                box-shadow: 0 16px 38px rgba(15, 23, 42, 0.06);
            }
            .hero-title { margin: 0; color: #0f172a; font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }
            .hero-copy { margin: 0.65rem 0 0 0; color: #475569; line-height: 1.7; }
            .section-card {
                display: none;
            }
            .section-block {
                margin-top: 10px;
                margin-bottom: 10px;
            }
            .section-heading {
                margin: 0 0 0.3rem 0;
                color: #0f172a;
                font-size: 1rem;
                font-weight: 800;
                letter-spacing: -0.02em;
            }
            .section-copy {
                margin: 0 0 0.85rem 0;
                color: #64748b;
                font-size: 0.92rem;
                line-height: 1.6;
            }
            .summary-box {
                margin-top: 10px; padding: 1rem 1.05rem; border-radius: 18px;
                border: 1px solid #e2e8f0; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            }
            .summary-box p { margin: 0.2rem 0; color: #0f172a; }
            .slot-title { margin: 0 0 0.5rem 0; color: #111827; font-size: 1rem; font-weight: 700; }
            .tab-shell {
                margin-top: 10px;
                margin-bottom: 10px;
                padding: 0.4rem;
                border-radius: 18px;
                border: 1px solid rgba(15, 23, 42, 0.06);
                background: rgba(255, 255, 255, 0.75);
            }
            .kpi {
                background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
                color: #fff; border-radius: 18px; padding: 1rem 1.1rem; box-shadow: 0 10px 28px rgba(17,24,39,0.18);
            }
            .kpi-label { font-size: 0.82rem; opacity: 0.82; }
            .kpi-value { font-size: 1.8rem; font-weight: 800; margin-top: 0.2rem; }
            .kpi-sub { font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem; }
            .insight-box { padding: 1rem; border-radius: 16px; background: #f8fafc; border: 1px dashed #cbd5e1; }
            .inline-note {
                color: #64748b;
                font-size: 0.88rem;
                margin-top: 10px;
            }
            .field-label {
                margin: 0 0 10px 0;
                color: #334155;
                font-size: 0.88rem;
                font-weight: 700;
            }
            .section-divider {
                height: 1px;
                margin: 10px 0;
                background: linear-gradient(90deg, rgba(203,213,225,0.9) 0%, rgba(203,213,225,0.35) 100%);
            }
            div[data-testid="stHorizontalBlock"] { gap: 10px; align-items: stretch; }
            div[data-testid="element-container"] {
                margin-bottom: 10px;
            }
            div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
                min-height: 2.8rem; border-radius: 14px; border: 1px solid #d1d5db; background: #fff; font-weight: 700;
            }
            div[data-testid="stButton"] > button[kind="primary"] { background: #111827; color: #fff; border: none; }
            div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div {
                border-radius: 14px; background: #fff;
            }
            div[data-testid="stRadio"] > label, div[data-testid="stNumberInput"] > label, div[data-testid="stTextInput"] > label {
                font-weight: 600;
                color: #334155;
            }
            @media (max-width: 640px) {
                .block-container {
                    padding-top: 1rem;
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                }
                .dashboard-card, .slot-card {
                    padding: 0.95rem;
                    border-radius: 18px;
                }
                .hero-title {
                    font-size: 1.6rem;
                }
                .hero-copy, .section-copy {
                    font-size: 0.9rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_slot_key(base: str, slot_id: int) -> str:
    return f"{base}_slot_{slot_id}"


def get_slot_app(slot_id: int):
    apps = st.session_state.selected_apps
    return apps[slot_id - 1] if 1 <= slot_id <= len(apps) else None


def get_slot_dataframe(slot_id: int):
    uploaded_df = st.session_state.get(get_slot_key("uploaded_df", slot_id))
    if uploaded_df is not None:
        return uploaded_df
    return st.session_state.get(get_slot_key("reviews_df", slot_id))


def switch_tab(tab_name: str) -> None:
    st.session_state.current_tab = tab_name


def handle_app_search() -> None:
    keyword = st.session_state.search_keyword.strip()
    st.session_state.crawler_search_error = ""
    st.session_state.search_results = []
    st.session_state.selected_candidate_index = 0
    st.session_state.selected_app_name = ""
    st.session_state.app_id_input = ""
    if not keyword:
        st.session_state.crawler_search_error = "앱 이름을 먼저 입력해 주세요."
        return
    try:
        candidates = search_apps(keyword, country_label=st.session_state.selected_country_label, limit=5)
    except Exception as error:
        st.session_state.crawler_search_error = f"검색 중 문제가 생겼어요: {error}"
        return
    if not candidates:
        st.session_state.crawler_search_error = "검색 결과를 찾지 못했어요. 다른 앱 이름이나 원어 이름으로 다시 시도해 주세요."
        return
    st.session_state.search_results = candidates
    st.session_state.selected_app_name = candidates[0]["title"]
    st.session_state.app_id_input = candidates[0]["appId"]


def handle_candidate_change() -> None:
    candidates = st.session_state.search_results
    index = int(st.session_state.selected_candidate_index)
    if 0 <= index < len(candidates):
        st.session_state.selected_app_name = candidates[index]["title"]
        st.session_state.app_id_input = candidates[index]["appId"]


def add_selected_app(candidate: dict) -> None:
    if len(st.session_state.selected_apps) >= MAX_SELECTED_APPS:
        st.session_state.crawler_search_error = "앱 리스트에는 최대 2개까지만 담을 수 있어요."
        return
    if any(item["appId"] == candidate["appId"] for item in st.session_state.selected_apps):
        st.session_state.crawler_search_error = "이미 리스트에 추가된 앱이에요."
        return
    st.session_state.selected_apps = st.session_state.selected_apps + [candidate]
    st.session_state.crawler_success_message = f"{candidate['title']} 앱을 리스트에 담았어요."


def handle_add_from_search() -> None:
    if not st.session_state.search_results:
        st.session_state.crawler_search_error = "먼저 검색 결과에서 앱을 찾은 뒤 추가해 주세요."
        return
    add_selected_app(st.session_state.search_results[int(st.session_state.selected_candidate_index)])


def handle_add_from_app_id() -> None:
    app_id_value = st.session_state.app_id_input.strip()
    if not app_id_value:
        st.session_state.crawler_search_error = "app ID를 먼저 입력해 주세요."
        return
    try:
        app_info = get_app_metadata(app_id_value, country_label=st.session_state.selected_country_label)
    except Exception as error:
        st.session_state.crawler_search_error = f"app ID 확인 중 문제가 생겼어요: {error}"
        return
    if not app_info:
        st.session_state.crawler_search_error = "입력한 app ID에 해당하는 앱을 찾지 못했어요."
        return
    add_selected_app(app_info)


def remove_app_from_list(slot_id: int) -> None:
    apps = st.session_state.selected_apps
    if not (1 <= slot_id <= len(apps)):
        return
    st.session_state.selected_apps = [item for index, item in enumerate(apps, start=1) if index != slot_id]
    old_state = {}
    for sid in SLOT_IDS:
        old_state[sid] = {
            "reviews_df": st.session_state.get(get_slot_key("reviews_df", sid)),
            "uploaded_df": st.session_state.get(get_slot_key("uploaded_df", sid)),
            "uploaded_name": st.session_state.get(get_slot_key("uploaded_name", sid)),
            "show_table": st.session_state.get(get_slot_key("show_table", sid), False),
            "review_count": st.session_state.get(get_slot_key("review_count", sid), 50),
        }
    for sid in SLOT_IDS:
        st.session_state[get_slot_key("reviews_df", sid)] = None
        st.session_state[get_slot_key("uploaded_df", sid)] = None
        st.session_state[get_slot_key("uploaded_name", sid)] = ""
        st.session_state[get_slot_key("show_table", sid)] = False
        st.session_state[get_slot_key("review_count", sid)] = 50
    target_sid = 1
    for old_sid in SLOT_IDS:
        if old_sid == slot_id or old_state[old_sid]["reviews_df"] is None and old_state[old_sid]["uploaded_df"] is None:
            continue
        st.session_state[get_slot_key("reviews_df", target_sid)] = old_state[old_sid]["reviews_df"]
        st.session_state[get_slot_key("uploaded_df", target_sid)] = old_state[old_sid]["uploaded_df"]
        st.session_state[get_slot_key("uploaded_name", target_sid)] = old_state[old_sid]["uploaded_name"]
        st.session_state[get_slot_key("show_table", target_sid)] = old_state[old_sid]["show_table"]
        st.session_state[get_slot_key("review_count", target_sid)] = old_state[old_sid]["review_count"]
        target_sid += 1


def handle_fetch_reviews_for_slot(slot_id: int) -> None:
    app_info = get_slot_app(slot_id)
    st.session_state.crawler_fetch_error = ""
    st.session_state.crawler_success_message = ""
    if app_info is None:
        st.session_state.crawler_fetch_error = "먼저 앱 리스트에 앱을 담아 주세요."
        return
    try:
        reviews_df = fetch_google_play_reviews(app_info["appId"], int(st.session_state[get_slot_key("review_count", slot_id)]))
    except Exception as error:
        st.session_state.crawler_fetch_error = f"리뷰를 가져오는 중 문제가 생겼어요: {error}"
        return
    if reviews_df.empty:
        st.session_state.crawler_fetch_error = "가져올 수 있는 리뷰를 찾지 못했어요."
        return
    st.session_state[get_slot_key("reviews_df", slot_id)] = reviews_df
    st.session_state[get_slot_key("uploaded_df", slot_id)] = None
    st.session_state[get_slot_key("uploaded_name", slot_id)] = ""
    st.session_state.crawler_success_message = f"{app_info['title']} 앱에서 총 {len(reviews_df)}개의 리뷰를 불러왔어요."


def toggle_slot_table(slot_id: int) -> None:
    key = get_slot_key("show_table", slot_id)
    st.session_state[key] = not st.session_state[key]


def load_uploaded_csv(slot_id: int) -> None:
    uploaded_file = st.session_state.get(f"analysis_upload_widget_{slot_id}")
    if uploaded_file is None:
        return
    try:
        st.session_state[get_slot_key("uploaded_df", slot_id)] = pd.read_csv(uploaded_file)
        st.session_state[get_slot_key("uploaded_name", slot_id)] = uploaded_file.name
        st.session_state.analysis_error = ""
    except Exception as error:
        st.session_state.analysis_error = f"파일 {slot_id} CSV를 읽는 중 문제가 생겼어요: {error}"


def render_dashboard_header() -> None:
    st.markdown(
        """
        <div class="dashboard-card">
            <h1 class="hero-title">Review Dashboard</h1>
            <p class="hero-copy">최대 2개 앱의 Google Play 리뷰를 각각 수집하고, 개별 분석과 비교 분석까지 한 화면에서 이어서 확인할 수 있는 통합 대시보드입니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab_navigation() -> None:
    active_tab = st.session_state.current_tab
    st.markdown("<div class='tab-shell'>", unsafe_allow_html=True)
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        st.button("Review Crawler", use_container_width=True, type="primary" if active_tab == CRAWLER_TAB else "secondary", on_click=switch_tab, args=(CRAWLER_TAB,))
    with col2:
        st.button("Review Analysis", use_container_width=True, type="primary" if active_tab == ANALYSIS_TAB else "secondary", on_click=switch_tab, args=(ANALYSIS_TAB,))
    st.markdown("</div>", unsafe_allow_html=True)


def render_selected_app_slot(slot_id: int) -> None:
    app_info = get_slot_app(slot_id)
    if app_info is None:
        st.markdown(f"<div class='slot-card'><p class='slot-title'>앱 슬롯 {slot_id}</p><p>아직 선택된 앱이 없어요.</p></div>", unsafe_allow_html=True)
        return
    reviews_df = st.session_state.get(get_slot_key("reviews_df", slot_id))
    st.markdown(
        f"<div class='slot-card'><p class='slot-title'>앱 슬롯 {slot_id}</p><p><strong>{app_info['title']}</strong></p><p>app ID: {app_info['appId']}</p></div>",
        unsafe_allow_html=True,
    )
    h1, h2, h3 = st.columns([1.2, 1.1, 0.9])
    with h1:
        st.markdown("<p class='field-label'>리뷰 개수</p>", unsafe_allow_html=True)
    with h2:
        st.markdown("<p class='field-label'>리뷰 수집</p>", unsafe_allow_html=True)
    with h3:
        st.markdown("<p class='field-label'>목록 관리</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1.1, 0.9])
    with c1:
        st.number_input(f"리뷰 개수 {slot_id}", key=get_slot_key("review_count", slot_id), min_value=1, max_value=MAX_REVIEW_COUNT, step=10, label_visibility="collapsed")
    with c2:
        st.button(f"리뷰 수집 {slot_id}", use_container_width=True, type="primary", on_click=handle_fetch_reviews_for_slot, args=(slot_id,))
    with c3:
        st.button(f"리스트 제거 {slot_id}", use_container_width=True, on_click=remove_app_from_list, args=(slot_id,))
    if reviews_df is not None and not reviews_df.empty:
        st.markdown(f"<div class='summary-box'><p><strong>총 {len(reviews_df)}개의 리뷰를 불러왔어요.</strong></p><p>앱 ID: {app_info['appId']}</p></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            st.button(f"바로보기 {slot_id}", use_container_width=True, on_click=toggle_slot_table, args=(slot_id,))
        with b2:
            st.download_button(f"CSV 다운로드 {slot_id}", convert_df_to_csv(reviews_df), file_name=f"{app_info['appId']}_reviews.csv", mime="text/csv", use_container_width=True)
        if st.session_state.get(get_slot_key("show_table", slot_id), False):
            st.dataframe(reviews_df[["작성자명", "리뷰 내용", "평점", "작성일", "앱 버전"]], use_container_width=True, hide_index=True)


def render_crawler_tab() -> None:
    st.markdown("<div class='section-block'><p class='section-heading'>앱 검색과 추가</p><p class='section-copy'>국가를 고른 뒤 앱 이름으로 검색하거나 app ID를 직접 입력해 비교용 리스트를 구성해보세요.</p></div>", unsafe_allow_html=True)
    h1, h2, h3 = st.columns([1.1, 3.4, 1.2])
    with h1:
        st.markdown("<p class='field-label'>국가</p>", unsafe_allow_html=True)
    with h2:
        st.markdown("<p class='field-label'>앱 이름 검색</p>", unsafe_allow_html=True)
    with h3:
        st.markdown("<p class='field-label'>검색 실행</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 3.4, 1.2])
    with c1:
        st.selectbox("국가 선택", options=list(COUNTRY_OPTIONS.keys()), key="selected_country_label", label_visibility="collapsed")
    with c2:
        st.text_input("앱 이름 검색", key="search_keyword", placeholder="예: 카카오톡, 네이버, Instagram", label_visibility="collapsed")
    with c3:
        st.button("앱 찾기", use_container_width=True, on_click=handle_app_search)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    h4, h5 = st.columns([3.6, 1.2])
    with h4:
        st.markdown("<p class='field-label'>Google Play app ID</p>", unsafe_allow_html=True)
    with h5:
        st.markdown("<p class='field-label'>리스트 추가</p>", unsafe_allow_html=True)

    c4, c5 = st.columns([3.6, 1.2])
    with c4:
        st.text_input("Google Play 앱 ID", key="app_id_input", placeholder="예: com.kakao.talk", label_visibility="collapsed")
    with c5:
        st.button("app ID 추가", use_container_width=True, on_click=handle_add_from_app_id)

    if st.session_state.crawler_search_error:
        st.warning(st.session_state.crawler_search_error)
    if st.session_state.crawler_fetch_error:
        st.warning(st.session_state.crawler_fetch_error)
    if st.session_state.crawler_success_message:
        st.success(st.session_state.crawler_success_message)

    if st.session_state.search_results:
        indexes = list(range(len(st.session_state.search_results)))
        st.markdown("<div class='section-block'><p class='section-heading'>검색 결과</p><p class='section-copy'>상위 5개 후보 중 원하는 앱을 고른 뒤 리스트에 추가해 주세요.</p></div>", unsafe_allow_html=True)
        st.radio(
            "검색 결과 상위 5개",
            options=indexes,
            key="selected_candidate_index",
            format_func=lambda i: f"{st.session_state.search_results[i]['title']}  |  {st.session_state.search_results[i]['appId']}",
            on_change=handle_candidate_change,
        )
        a1, a2 = st.columns([1.2, 4])
        with a1:
            st.button("리스트에 추가", use_container_width=True, type="primary", on_click=handle_add_from_search)
        with a2:
            if st.session_state.selected_app_name and st.session_state.app_id_input:
                st.markdown(
                    f"<div class='summary-box'><p><strong>선택된 앱</strong></p><p>{st.session_state.selected_app_name}</p><p>app ID: {st.session_state.app_id_input}</p></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div class='section-block'><p class='section-heading'>선택한 앱 리스트</p><p class='section-copy'>각 슬롯에서 리뷰 개수를 정하고 수집을 실행해 주세요.</p></div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        render_selected_app_slot(1)
    with s2:
        render_selected_app_slot(2)

    if any(st.session_state.get(get_slot_key("reviews_df", sid)) is not None for sid in SLOT_IDS):
        if st.button("리뷰 분석하기", use_container_width=True, type="primary"):
            st.session_state.current_tab = ANALYSIS_TAB

def render_kpi_cards(df_work: pd.DataFrame) -> None:
    total_reviews = len(df_work)
    avg_score = float(df_work["score"].mean()) if df_work["score"].notna().any() else None
    avg_score_text = f"{avg_score:.2f}" if avg_score is not None else "-"
    valid_dates = df_work["date"].dropna()
    period_text = f"{valid_dates.min().date()} ~ {valid_dates.max().date()}" if len(valid_dates) > 0 else "작성일 정보 없음"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>전체 리뷰 수</div><div class='kpi-value'>{total_reviews:,}</div><div class='kpi-sub'>현재 분석 대상 전체 행 기준</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>평균 평점</div><div class='kpi-value'>{avg_score_text}</div><div class='kpi-sub'>리뷰 평점 컬럼 기준</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>리뷰 작성 기간</div><div class='kpi-value' style='font-size:1rem;'>{period_text}</div><div class='kpi-sub'>작성일 컬럼 기준</div></div>", unsafe_allow_html=True)


def filter_dataframe_by_date_range(dataframe: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if dataframe is None or dataframe.empty or "date" not in dataframe.columns:
        return dataframe
    if start_date is None or end_date is None:
        return dataframe
    filtered = dataframe.copy()
    filtered = filtered.dropna(subset=["date"])
    if filtered.empty:
        return filtered
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return filtered[(filtered["date"] >= start_ts) & (filtered["date"] <= end_ts)]


def render_single_analysis(dataframe: pd.DataFrame, label: str, granularity_key: str) -> None:
    cleaning_key = "use_cleaning_file_1" if granularity_key == "analysis_file_1_granularity" else "use_cleaning_file_2"
    df_work = normalize_analysis_dataframe(dataframe)
    if df_work is None or df_work.empty:
        st.info(f"{label} 데이터가 비어 있어 분석을 진행할 수 없어요.")
        return
    use_cleaning = st.checkbox("클렌징 적용", key=cleaning_key, help="감탄사, 강조어, 반복 표현처럼 정보량이 낮은 단어를 줄여서 키워드와 워드클라우드를 더 또렷하게 보여줍니다.")
    st.markdown(f"#### {label} 개요")
    render_kpi_cards(df_work)
    st.markdown("#### 기간별 리뷰 수 추이")
    if df_work["date"].notna().any():
        st.line_chart(build_daily_trend(df_work)["리뷰수"], use_container_width=True)
    else:
        st.info("작성일 정보를 확인할 수 없어 기간별 추이 차트는 잠시 건너뛸게요.")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("#### 평점 분포")
        if df_work["score"].notna().any():
            st.bar_chart(build_score_distribution(df_work)["count"], use_container_width=True)
        else:
            st.info("평점 정보가 부족해요.")
    with c2:
        st.markdown("#### 핵심 키워드")
        keywords = top_keywords_from_reviews(df_work["review"], top_n=20, use_cleaning=use_cleaning)
        if keywords:
            st.dataframe(pd.DataFrame(keywords, columns=["단어", "빈도"]), use_container_width=True, hide_index=True)
        else:
            st.info("키워드를 추출할 텍스트가 아직 충분하지 않아요.")
    st.markdown("#### 기간별 키워드")
    if df_work["date"].notna().any():
        granularity = st.selectbox("기간 단위", ["일(Day)", "주(Week)", "월(Month)"], index=["일(Day)", "주(Week)", "월(Month)"].index(st.session_state[granularity_key]), key=granularity_key)
        st.dataframe(build_period_keyword_table(df_work, granularity, use_cleaning=use_cleaning), use_container_width=True, hide_index=True)
    else:
        st.info("작성일과 리뷰 내용이 함께 있어야 기간별 키워드를 보여드릴 수 있어요.")
    st.markdown("#### 긍정/부정 워드클라우드")
    w1, w2 = st.columns(2)
    with w1:
        st.caption("긍정 리뷰: score >= 4")
        fig = make_wordcloud(" ".join(df_work[df_work["score"] >= 4]["review"].fillna("").astype(str).tolist()), use_cleaning=use_cleaning)
        if fig is not None:
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.info("긍정 리뷰 텍스트가 충분하지 않아요.")
    with w2:
        st.caption("부정 리뷰: score <= 2")
        fig = make_wordcloud(" ".join(df_work[df_work["score"] <= 2]["review"].fillna("").astype(str).tolist()), use_cleaning=use_cleaning)
        if fig is not None:
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.info("부정 리뷰 텍스트가 충분하지 않아요.")
    st.markdown("#### 감성 분포와 인사이트")
    s1, s2 = st.columns([1, 1.2])
    with s1:
        sentiment_df = build_sentiment_distribution(df_work)
        st.bar_chart(sentiment_df.set_index("감성")["개수"], use_container_width=True)
    with s2:
        top_words = [word for word, _ in top_keywords_from_reviews(df_work["review"], top_n=5, use_cleaning=use_cleaning)]
        lines = [f"전체 리뷰 수: {len(df_work):,}건"]
        if df_work["score"].notna().any():
            lines.append(f"평균 평점: {df_work['score'].mean():.2f}점")
        if df_work["date"].notna().any():
            valid_dates = df_work["date"].dropna()
            lines.append(f"작성 기간: {valid_dates.min().date()} ~ {valid_dates.max().date()}")
        if top_words:
            lines.append(f"핵심 키워드: {', '.join(top_words[:3])}")
        st.markdown("<div class='insight-box'>" + "<br/>".join([f"• {line}" for line in lines]) + "</div>", unsafe_allow_html=True)


def render_compare_analysis(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> None:
    use_cleaning = st.checkbox("비교 분석에 클렌징 적용", key="use_cleaning_compare", help="핵심 키워드와 워드 기준 비교에서 의미가 약한 표현을 줄입니다.")
    normalized_df1 = normalize_analysis_dataframe(df1)
    normalized_df2 = normalize_analysis_dataframe(df2)
    compare_df1 = normalized_df1
    compare_df2 = normalized_df2

    if (
        normalized_df1 is not None
        and normalized_df2 is not None
        and not normalized_df1.empty
        and not normalized_df2.empty
        and normalized_df1["date"].notna().any()
        and normalized_df2["date"].notna().any()
    ):
        min_date = min(normalized_df1["date"].dropna().min().date(), normalized_df2["date"].dropna().min().date())
        max_date = max(normalized_df1["date"].dropna().max().date(), normalized_df2["date"].dropna().max().date())
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            start_date = st.date_input("비교 시작일", value=min_date, min_value=min_date, max_value=max_date, key="compare_start_date")
        with range_col2:
            end_date = st.date_input("비교 종료일", value=max_date, min_value=min_date, max_value=max_date, key="compare_end_date")
        if start_date > end_date:
            st.warning("비교 시작일이 종료일보다 늦어요. 날짜를 다시 확인해 주세요.")
            return
        compare_df1 = filter_dataframe_by_date_range(normalized_df1, start_date, end_date)
        compare_df2 = filter_dataframe_by_date_range(normalized_df2, start_date, end_date)
        if compare_df1.empty or compare_df2.empty:
            st.info("선택한 기간에 비교할 리뷰가 충분하지 않아요. 기간 범위를 조금 넓혀 주세요.")
            return

    st.markdown("#### 평점 비교")
    st.dataframe(compare_scores(compare_df1, compare_df2, label1, label2), use_container_width=True, hide_index=True)
    st.markdown("#### 핵심 키워드 비교")
    st.dataframe(compare_keywords(compare_df1, compare_df2, label1, label2, top_n=10, use_cleaning=use_cleaning), use_container_width=True, hide_index=True)
    st.markdown("#### 리뷰 긍정도 비교")
    sentiment_df = compare_sentiment(compare_df1, compare_df2, label1, label2)
    st.bar_chart(sentiment_df.set_index("파일"), use_container_width=True)
    st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
    st.markdown("#### 동향 차이")
    granularity = st.selectbox("비교 기간 단위", ["일(Day)", "주(Week)", "월(Month)"], index=["일(Day)", "주(Week)", "월(Month)"].index(st.session_state.analysis_compare_granularity), key="analysis_compare_granularity")
    trend_df = compare_trends(compare_df1, compare_df2, label1, label2, granularity)
    st.line_chart(trend_df.set_index("period"), use_container_width=True)
    st.dataframe(trend_df, use_container_width=True, hide_index=True)
    st.markdown("#### 비교 요약")
    summary_lines = build_compare_summary(compare_df1, compare_df2, label1, label2, use_cleaning=use_cleaning)
    st.markdown("<div class='insight-box'>" + "<br/>".join([f"• {line}" for line in summary_lines]) + "</div>", unsafe_allow_html=True)


def render_analysis_source_status(slot_id: int) -> str:
    if st.session_state.get(get_slot_key("uploaded_name", slot_id)):
        return f"업로드 파일: {st.session_state[get_slot_key('uploaded_name', slot_id)]}"
    app_info = get_slot_app(slot_id)
    if st.session_state.get(get_slot_key("reviews_df", slot_id)) is not None and app_info is not None:
        return f"Crawler 데이터: {app_info['title']}"
    return "데이터 없음"


def render_analysis_tab() -> None:
    st.markdown("<div class='section-block'><p class='section-heading'>분석 데이터 준비</p><p class='section-copy'>Crawler에서 수집한 데이터는 자동으로 연결되고, 필요하면 각 슬롯에 CSV를 직접 올릴 수 있어요.</p></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-card">
            <p class="section-heading">분석 데이터 준비</p>
            <p class="section-copy">Crawler에서 수집한 데이터는 자동으로 연결되고, 필요하면 각 슬롯에 CSV를 직접 업로드해 덮어쓸 수 있어요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("<p class='field-label'>파일 1 업로드</p>", unsafe_allow_html=True)
    with a2:
        st.markdown("<p class='field-label'>파일 2 업로드</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.file_uploader("파일 1 CSV 업로드", type=["csv"], key="analysis_upload_widget_1", on_change=load_uploaded_csv, args=(1,))
        st.markdown(f"<p class='inline-note'>{render_analysis_source_status(1)}</p>", unsafe_allow_html=True)
    with c2:
        st.file_uploader("파일 2 CSV 업로드", type=["csv"], key="analysis_upload_widget_2", on_change=load_uploaded_csv, args=(2,))
        st.markdown(f"<p class='inline-note'>{render_analysis_source_status(2)}</p>", unsafe_allow_html=True)
    if st.session_state.analysis_error:
        st.warning(st.session_state.analysis_error)
    compare_tab, file_1_tab, file_2_tab = st.tabs(["파일 비교", "개별 파일 1", "개별 파일 2"])
    df1 = get_slot_dataframe(1)
    df2 = get_slot_dataframe(2)
    label1 = get_slot_app(1)["title"] if get_slot_app(1) is not None else "파일 1"
    label2 = get_slot_app(2)["title"] if get_slot_app(2) is not None else "파일 2"
    with compare_tab:
        if df1 is None or df2 is None:
            st.info("비교 분석을 보려면 두 개의 데이터셋이 모두 필요해요.")
        else:
            render_compare_analysis(df1, df2, label1, label2)
    with file_1_tab:
        if df1 is None:
            st.info("파일 1 데이터가 아직 없어요.")
        else:
            render_single_analysis(df1, label1, "analysis_file_1_granularity")
    with file_2_tab:
        if df2 is None:
            st.info("파일 2 데이터가 아직 없어요.")
        else:
            render_single_analysis(df2, label2, "analysis_file_2_granularity")


disable_broken_proxy_settings()
init_session_state()
render_styles()
render_dashboard_header()
render_tab_navigation()

if st.session_state.current_tab == CRAWLER_TAB:
    render_crawler_tab()
else:
    render_analysis_tab()
