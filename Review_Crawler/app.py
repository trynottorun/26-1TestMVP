from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from review_core.crawler import COUNTRY_OPTIONS, fetch_google_play_reviews, get_app_metadata, search_apps
from review_core.shared import convert_df_to_csv, disable_broken_proxy_settings


st.set_page_config(page_title="Google Play 리뷰 수집기", page_icon="G", layout="centered")

PERIOD_OPTIONS = [
    ("all", "전체 기간"),
    ("1m", "최근 1개월"),
    ("2m", "최근 2개월"),
    ("3m", "최근 3개월"),
]


def init_session_state() -> None:
    defaults = {
        "selected_country_label": "한국",
        "search_keyword": "",
        "direct_app_id": "",
        "search_results": [],
        "selected_candidate_index": 0,
        "selected_app_name": "",
        "selected_package_id": "",
        "selected_period": "all",
        "review_count": 50,
        "reviews_df": None,
        "last_loaded_count": 0,
        "show_table": False,
        "search_error": "",
        "fetch_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_review_state() -> None:
    st.session_state.reviews_df = None
    st.session_state.last_loaded_count = 0
    st.session_state.show_table = False
    st.session_state.fetch_error = ""


def handle_app_search() -> None:
    keyword = st.session_state.search_keyword.strip()
    st.session_state.search_error = ""
    st.session_state.search_results = []
    st.session_state.selected_candidate_index = 0
    st.session_state.selected_app_name = ""
    st.session_state.selected_package_id = ""
    reset_review_state()
    if not keyword:
        st.session_state.search_error = "앱 이름을 먼저 입력해 주세요."
        return
    try:
        candidates = search_apps(keyword, country_label=st.session_state.selected_country_label, limit=5)
    except Exception as error:
        st.session_state.search_error = f"검색 중 문제가 생겼어요: {error}"
        return
    if not candidates:
        st.session_state.search_error = "검색 결과를 찾지 못했어요. 다른 앱 이름이나 원어 이름으로 다시 시도해 주세요."
        return
    st.session_state.search_results = candidates
    st.session_state.selected_app_name = candidates[0]["title"]
    st.session_state.selected_package_id = candidates[0]["appId"]
    st.session_state.direct_app_id = candidates[0]["appId"]


def handle_direct_app_id_apply() -> None:
    direct_app_id = st.session_state.direct_app_id.strip()
    st.session_state.search_error = ""
    st.session_state.search_results = []
    st.session_state.selected_candidate_index = 0
    st.session_state.selected_app_name = ""
    st.session_state.selected_package_id = ""
    reset_review_state()
    if not direct_app_id:
        st.session_state.search_error = "appID를 먼저 입력해 주세요."
        return
    try:
        app_info = get_app_metadata(direct_app_id, country_label=st.session_state.selected_country_label)
    except Exception as error:
        st.session_state.search_error = f"appID 확인 중 문제가 생겼어요: {error}"
        return
    if not app_info:
        st.session_state.search_error = "입력한 appID에 해당하는 앱을 찾지 못했어요."
        return
    st.session_state.search_results = [app_info]
    st.session_state.selected_app_name = app_info["title"]
    st.session_state.selected_package_id = app_info["appId"]
    st.session_state.direct_app_id = app_info["appId"]


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
        st.session_state.fetch_error = "먼저 앱을 검색하고 결과에서 하나를 선택해 주세요."
        return
    try:
        reviews_df = fetch_google_play_reviews(package_id, review_count, period_key=period_key)
    except Exception as error:
        st.session_state.fetch_error = f"리뷰를 불러오는 중 문제가 생겼어요: {error}"
        return
    if reviews_df.empty:
        st.session_state.fetch_error = "선택한 조건에 맞는 리뷰를 찾지 못했어요."
        return
    st.session_state.reviews_df = reviews_df
    st.session_state.last_loaded_count = len(reviews_df)


disable_broken_proxy_settings()
init_session_state()

st.title("Google Play 리뷰 수집기")
st.caption("앱 이름으로 검색하거나 appID를 직접 입력해 원하는 앱의 리뷰를 수집합니다.")

country_col, search_col, button_col = st.columns([1.4, 3.2, 1.2])
with country_col:
    st.selectbox("국가 선택", options=list(COUNTRY_OPTIONS.keys()), key="selected_country_label", label_visibility="collapsed")
with search_col:
    st.text_input("앱 이름 입력", key="search_keyword", placeholder="예: 카카오톡, 네이버, Instagram", label_visibility="collapsed")
with button_col:
    st.button("앱 찾기", type="primary", use_container_width=True, on_click=handle_app_search)

direct_app_col, direct_button_col = st.columns([4, 1.2])
with direct_app_col:
    st.text_input("appID 직접 입력", key="direct_app_id", placeholder="예: com.kakao.talk", label_visibility="collapsed")
with direct_button_col:
    st.button("appID 적용", use_container_width=True, on_click=handle_direct_app_id_apply)

if st.session_state.search_error:
    st.warning(st.session_state.search_error)

if st.session_state.search_results:
    option_indexes = list(range(len(st.session_state.search_results)))
    st.radio(
        "검색 결과 상위 5개",
        options=option_indexes,
        key="selected_candidate_index",
        format_func=lambda index: f"{st.session_state.search_results[index]['title']}  |  {st.session_state.search_results[index]['appId']}",
        on_change=handle_candidate_change,
    )
    st.info(f"선택된 앱: {st.session_state.selected_app_name} / appID: {st.session_state.selected_package_id}")

st.number_input("불러올 리뷰 개수", key="review_count", min_value=1, max_value=50000, value=50, step=10)
period_columns = st.columns(len(PERIOD_OPTIONS))
for index, option in enumerate(PERIOD_OPTIONS):
    period_key, period_label = option
    with period_columns[index]:
        st.button(period_label, key=f"period_{period_key}", type="primary" if st.session_state.selected_period == period_key else "secondary", use_container_width=True, on_click=handle_period_change, args=(period_key,))

st.button("리뷰 불러오기", type="primary", use_container_width=True, on_click=handle_fetch_reviews)

if st.session_state.fetch_error:
    st.warning(st.session_state.fetch_error)

if st.session_state.reviews_df is not None and not st.session_state.reviews_df.empty:
    st.success(f"총 {st.session_state.last_loaded_count}개의 리뷰를 불러왔어요.")
    preview_col, download_col = st.columns(2)
    with preview_col:
        if st.button("바로보기", use_container_width=True):
            st.session_state.show_table = not st.session_state.show_table
    with download_col:
        st.download_button("CSV 다운로드", convert_df_to_csv(st.session_state.reviews_df), file_name="google_play_reviews.csv", mime="text/csv", use_container_width=True)
    if st.session_state.show_table:
        st.dataframe(st.session_state.reviews_df, use_container_width=True, hide_index=True)
