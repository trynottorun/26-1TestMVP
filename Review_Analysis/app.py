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


st.set_page_config(page_title="분석 리포트 대시보드", layout="wide")

theme_base = st.get_option("theme.base")
is_dark = theme_base == "dark"

st.title("분석 리포트 대시보드")
st.caption("CSV 리뷰 데이터를 업로드하고, [분석 시작] 버튼으로 기본 분석 리포트를 생성합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("업로드 성공! CSV 파일을 DataFrame으로 읽었습니다.")
    except Exception as error:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {error}")
        df = None

    if df is not None:
        st.dataframe(df.head(5), use_container_width=True)
        use_cleaning = st.checkbox(
            "클렌징 적용",
            value=True,
            help="감탄사, 강조어, 반복 표현처럼 정보량이 낮은 단어를 줄여서 키워드와 워드클라우드를 더 또렷하게 보여줍니다.",
        )
        if st.button("분석 시작", type="primary", use_container_width=True):
            df_work = normalize_analysis_dataframe(df)
            if df_work is None or df_work.empty:
                st.info("분석할 수 있는 유효한 데이터가 없어요.")
            else:
                total_reviews = len(df_work)
                avg_score = float(df_work["score"].mean()) if df_work["score"].notna().any() else None
                valid_dates = df_work["date"].dropna()
                period_text = f"{valid_dates.min().date()} ~ {valid_dates.max().date()}" if len(valid_dates) > 0 else "작성일 정보 없음"

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("전체 리뷰 수", f"{total_reviews:,}")
                with k2:
                    st.metric("평균 평점", f"{avg_score:.2f}" if avg_score is not None else "-")
                with k3:
                    st.metric("리뷰 작성 기간", period_text)

                st.subheader("기간별 리뷰 수 추이")
                if df_work["date"].notna().any():
                    st.line_chart(build_daily_trend(df_work)["리뷰수"], use_container_width=True)
                else:
                    st.info("작성일 정보가 없어 추이 차트를 표시할 수 없습니다.")

                score_col, keyword_col = st.columns([1, 1.2])
                with score_col:
                    st.subheader("평점 분포")
                    if df_work["score"].notna().any():
                        st.bar_chart(build_score_distribution(df_work)["count"], use_container_width=True)
                    else:
                        st.info("평점 정보가 부족합니다.")
                with keyword_col:
                    st.subheader("핵심 키워드")
                    keywords = top_keywords_from_reviews(df_work["review"], top_n=20, use_cleaning=use_cleaning)
                    if keywords:
                        st.dataframe(pd.DataFrame(keywords, columns=["단어", "빈도"]), use_container_width=True, hide_index=True)
                    else:
                        st.info("키워드를 추출할 리뷰 텍스트가 없습니다.")

                st.subheader("기간별 키워드")
                if df_work["date"].notna().any():
                    granularity = st.selectbox("기간 단위", ["일(Day)", "주(Week)", "월(Month)"], index=1)
                    st.dataframe(build_period_keyword_table(df_work, granularity, use_cleaning=use_cleaning), use_container_width=True, hide_index=True)
                else:
                    st.info("작성일 정보가 없어 기간별 키워드를 표시할 수 없습니다.")

                positive_col, negative_col = st.columns(2)
                with positive_col:
                    st.subheader("긍정 워드클라우드")
                    positive_text = " ".join(df_work[df_work["score"] >= 4]["review"].fillna("").astype(str).tolist())
                    positive_fig = make_wordcloud(positive_text, is_dark_mode=is_dark, use_cleaning=use_cleaning)
                    if positive_fig is not None:
                        st.pyplot(positive_fig, clear_figure=True, use_container_width=True)
                    else:
                        st.info("긍정 리뷰 텍스트가 부족합니다.")
                with negative_col:
                    st.subheader("부정 워드클라우드")
                    negative_text = " ".join(df_work[df_work["score"] <= 2]["review"].fillna("").astype(str).tolist())
                    negative_fig = make_wordcloud(negative_text, is_dark_mode=is_dark, use_cleaning=use_cleaning)
                    if negative_fig is not None:
                        st.pyplot(negative_fig, clear_figure=True, use_container_width=True)
                    else:
                        st.info("부정 리뷰 텍스트가 부족합니다.")

                st.subheader("감성 분포")
                sentiment_df = build_sentiment_distribution(df_work)
                st.bar_chart(sentiment_df.set_index("감성")["개수"], use_container_width=True)
