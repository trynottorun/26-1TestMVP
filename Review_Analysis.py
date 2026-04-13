import streamlit as st
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from openai import OpenAI


# =========================
# 페이지 설정 / 스타일
# =========================
st.set_page_config(page_title="분석 리포트 대시보드", layout="wide")

theme_base = st.get_option("theme.base")
is_dark = theme_base == "dark"

st.markdown(
    """
<style>
  .app-wrap { max-width: 1200px; margin: 0 auto; }
  .muted { color: rgba(var(--text-color), 0.70); }
  .card {
    background: rgba(var(--background-color), 1);
    border: 1px solid rgba(var(--text-color), 0.10);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
  }
  .card-title { font-weight: 700; font-size: 16px; margin-bottom: 6px; }
  .card-desc { color: rgba(var(--text-color), 0.70); font-size: 13px; margin-bottom: 10px; }
  .kpi {
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    color: #fff;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 10px 28px rgba(17,24,39,0.25);
    border: 1px solid rgba(255,255,255,0.10);
  }
  .kpi .label { font-size: 13px; opacity: 0.85; }
  .kpi .value { font-size: 28px; font-weight: 800; margin-top: 2px; }
  .kpi .sub { font-size: 12px; opacity: 0.8; margin-top: 4px; }
  .section-gap { height: 10px; }
  .pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(59,130,246,0.12);
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 600;
  }
  .insight-box {
    background: rgba(var(--secondary-background-color), 1);
    border: 1px dashed rgba(var(--text-color), 0.22);
    border-radius: 14px;
    padding: 14px 16px;
  }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 상단 제목/설명
# =========================
st.markdown('<div class="app-wrap">', unsafe_allow_html=True)
st.title("분석 리포트 대시보드 (AI 해석 포함)")
st.caption("CSV 리뷰 데이터를 업로드하고, [분석 시작] 버튼으로 기본 분석 + AI 해석 리포트를 생성합니다.")
st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 업로드
# =========================
st.markdown('<div class="app-wrap">', unsafe_allow_html=True)
with st.container():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">1) 기본 정보</div>
          <div class="card-desc">CSV 파일을 업로드하면 파일 정보와 미리보기를 확인한 뒤 분석을 시작할 수 있습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def simple_tokenize(text: str):
    if not isinstance(text, str):
        return []
    cleaned = text.replace("\n", " ").replace("\r", " ").strip().lower()
    for ch in [
        ".",
        ",",
        "!",
        "?",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        ":",
        ";",
        "/",
        "\\",
        "|",
        "-",
        "_",
        "*",
        "+",
        "=",
        "#",
        "@",
    ]:
        cleaned = cleaned.replace(ch, " ")
    words = [w for w in cleaned.split() if len(w) >= 2]
    return words


def top_keywords_from_reviews(series: pd.Series, top_n: int = 10):
    stopwords = {
        "the",
        "and",
        "to",
        "of",
        "is",
        "in",
        "it",
        "this",
        "that",
        "for",
        "with",
        "you",
        "your",
        "i",
        "we",
        "a",
        "an",
    }

    all_words = []
    for t in series.fillna("").astype(str).tolist():
        for w in simple_tokenize(t):
            if w not in stopwords:
                all_words.append(w)

    c = Counter(all_words)
    return c.most_common(top_n)


def make_wordcloud(text: str, is_dark_mode: bool):
    bg = "#0b1220" if is_dark_mode else "#ffffff"
    wc = WordCloud(
        width=1200,
        height=700,
        background_color=bg,
        collocations=False,
        max_words=120,
        font_path="C:/Windows/Fonts/malgun.ttf",
    ).generate(text if text.strip() else " ")
    fig = plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig


# =========================
# OpenAI 헬퍼 (AI 해석용)
# =========================
def get_openai_client():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception:
        return None
    return OpenAI(api_key=api_key)


def ask_openai(prompt: str) -> str:
    client = get_openai_client()
    if client is None:
        return "OpenAI API 키가 설정되어 있지 않아 AI 해석을 생성할 수 없습니다."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 한국어로 짧고 명확하게 분석 결과를 요약해 주는 데이터 분석 리포트 보조 도구야. 불필요한 장식 없이 핵심만 설명해.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI 해석 생성 중 오류가 발생했습니다: {e}"


# =========================
# 업로드 후 처리 (분석 코드는 반드시 이 블록 안)
# =========================
if uploaded_file is not None:
    # -------------------------
    # 파일 읽기
    # -------------------------
    try:
        df = pd.read_csv(uploaded_file)
        st.success("업로드 성공! CSV 파일을 DataFrame으로 읽었습니다.")
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
        df = None

    if df is not None:
        # -------------------------
        # 파일 정보
        # -------------------------
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        info_col1, info_col2, info_col3 = st.columns([2.2, 1, 1])
        with info_col1:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">파일 정보</div>
                  <div class="card-desc muted">업로드한 파일의 기본 메타데이터입니다.</div>
                  <div><span class="pill">파일명</span> &nbsp; <b>{uploaded_file.name}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_col2:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">행(Row)</div>
                  <div class="card-desc muted">총 행 개수</div>
                  <div style="font-size:26px; font-weight:800;">{len(df):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_col3:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">열(Column)</div>
                  <div class="card-desc muted">총 열 개수</div>
                  <div style="font-size:26px; font-weight:800;">{len(df.columns):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------
        # 데이터 미리보기
        # -------------------------
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">데이터 미리보기</div>
                  <div class="card-desc">상위 5개 행을 확인하고, 컬럼명이 요구사항(score/date/review)과 맞는지 확인하세요.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(5), use_container_width=True)

        # -------------------------
        # 분석 시작 버튼
        # -------------------------
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        analyze_button = st.button("분석 시작", type="primary", use_container_width=True)

        # -------------------------
        # 분석 섹션 (버튼 클릭 시 출력)
        # -------------------------
        if analyze_button:
            # =====================================
            # 개요 분석
            # =====================================
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.header("2) 개요 분석")
            st.caption("리뷰 수/평균 평점/작성 기간을 KPI로 보고, 기간별 리뷰 추이를 확인합니다.")

            df_work = df.copy()
            has_score = "score" in df_work.columns
            has_date = "date" in df_work.columns
            has_text = "review" in df_work.columns

            if has_date:
                df_work["date"] = safe_to_datetime(df_work["date"])

            total_reviews = len(df_work)
            avg_score = float(df_work["score"].mean()) if has_score else None

            period_text = "date 컬럼 없음"
            if has_date:
                valid_dates = df_work["date"].dropna()
                if len(valid_dates) > 0:
                    start = valid_dates.min().date()
                    end = valid_dates.max().date()
                    period_text = f"{start} ~ {end}"
                else:
                    period_text = "date 파싱 실패"

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(
                    f"""
                    <div class="kpi">
                      <div class="label">전체 리뷰 수</div>
                      <div class="value">{total_reviews:,}</div>
                      <div class="sub">업로드된 전체 행 기준</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k2:
                val = f"{avg_score:.2f}" if avg_score is not None else "-"
                sub = "score 컬럼 기준" if has_score else "score 컬럼 없음"
                st.markdown(
                    f"""
                    <div class="kpi">
                      <div class="label">평균 평점</div>
                      <div class="value">{val}</div>
                      <div class="sub">{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k3:
                sub = "date 컬럼 기준" if has_date else "date 컬럼 없음"
                st.markdown(
                    f"""
                    <div class="kpi">
                      <div class="label">리뷰 작성 기간</div>
                      <div class="value" style="font-size:18px; line-height:1.2;">{period_text}</div>
                      <div class="sub">{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">기간별 리뷰 수 추이</div>
                      <div class="card-desc">date를 날짜형으로 변환한 뒤, 일자별 리뷰 건수를 라인 차트로 표시합니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if has_date and df_work["date"].notna().any():
                    daily = (
                        df_work.dropna(subset=["date"])
                        .assign(일자=lambda x: x["date"].dt.date)
                        .groupby("일자")
                        .size()
                        .reset_index(name="리뷰수")
                        .sort_values("일자")
                    )
                    daily = daily.set_index("일자")
                    st.line_chart(daily["리뷰수"], use_container_width=True)
                else:
                    st.info("date 컬럼이 없거나 날짜로 변환되지 않아 추이 차트를 표시할 수 없습니다.")

            # -------------------------
            # 평점 분포 + 평점별 키워드
            # -------------------------
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            dist_col1, dist_col2 = st.columns([1.0, 1.2])

            with dist_col1:
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">평점(score) 분포</div>
                      <div class="card-desc">1~5점 각 평점별 리뷰 개수를 막대 그래프로 표시합니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if has_score:
                    s = pd.to_numeric(df_work["score"], errors="coerce")
                    counts = s.dropna().astype(int).value_counts()
                    score_order = [1, 2, 3, 4, 5]
                    counts = counts.reindex(score_order).fillna(0).astype(int)
                    score_dist_df = pd.DataFrame({"count": counts.values}, index=score_order)
                    st.bar_chart(score_dist_df["count"], use_container_width=True)
                else:
                    st.info('score 컬럼("score")이 없어 평점 분포를 표시할 수 없습니다.')

            with dist_col2:
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">평점별 리뷰 키워드</div>
                      <div class="card-desc">각 평점(score)별로 review에서 자주 등장한 키워드를 TOP 5로 요약합니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if has_score and has_text:
                    score_keyword_rows = []
                    for sc in [1, 2, 3, 4, 5]:
                        group = df_work[df_work["score"] == sc]
                        if len(group) == 0:
                            score_keyword_rows.append({"score": sc, "reviews": 0, "top_keywords": "-"})
                        else:
                            top = top_keywords_from_reviews(group["review"], top_n=5)
                            kw = ", ".join([w for w, _ in top]) if len(top) > 0 else "-"
                            score_keyword_rows.append({"score": sc, "reviews": len(group), "top_keywords": kw})

                    score_kw_df = pd.DataFrame(score_keyword_rows)
                    st.dataframe(score_kw_df, use_container_width=True, hide_index=True)
                else:
                    if not has_score:
                        st.info('score 컬럼("score")이 없어 평점별 키워드를 만들 수 없습니다.')
                    elif not has_text:
                        st.info('review 컬럼("review")이 없어 평점별 키워드를 만들 수 없습니다.')

            # -------------------------
            # 기간별 score & review 키워드
            # -------------------------
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">기간별 score & review 키워드</div>
                  <div class="card-desc">기간 단위별로 review에서 특히 많이 나타나는 키워드를 TOP 3으로 정리합니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if has_date and has_text and df_work["date"].notna().any():
                gran = st.selectbox("기간 단위", ["일(Day)", "주(Week)", "월(Month)"], index=1)
                if gran.startswith("일"):
                    df_work["_period"] = df_work["date"].dt.date.astype(str)
                elif gran.startswith("주"):
                    df_work["_period"] = df_work["date"].dt.to_period("W").astype(str)
                else:
                    df_work["_period"] = df_work["date"].dt.to_period("M").astype(str)

                period_rows = []
                for p, g in df_work.dropna(subset=["date"]).groupby("_period"):
                    top = top_keywords_from_reviews(g["review"], top_n=3)
                    kw = ", ".join([w for w, _ in top]) if len(top) > 0 else "-"
                    avg_s = None
                    if has_score:
                        avg_s = pd.to_numeric(g["score"], errors="coerce").mean()
                    period_rows.append(
                        {
                            "period": p,
                            "reviews": len(g),
                            "avg_score": round(float(avg_s), 2) if avg_s == avg_s else None,
                            "top_keywords": kw,
                        }
                    )

                period_df = pd.DataFrame(period_rows).sort_values("period")
                st.dataframe(period_df, use_container_width=True, hide_index=True)
            else:
                st.info('date/review 컬럼이 없거나 date가 유효하지 않아 기간별 키워드를 만들 수 없습니다.')

            # -------------------------
            # 긍정/부정 워드클라우드
            # -------------------------
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.subheader("긍정(positive) 워드클라우드")
            st.caption("score >= 4 인 review 텍스트로 생성합니다.")
            if has_score and has_text:
                pos = df_work[pd.to_numeric(df_work["score"], errors="coerce") >= 4]
                pos_text = " ".join(pos["review"].fillna("").astype(str).tolist())
                if pos_text.strip():
                    fig = make_wordcloud(pos_text, is_dark_mode=is_dark)
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                else:
                    st.info("긍정 리뷰 텍스트가 없어 워드클라우드를 만들 수 없습니다.")
            else:
                st.info('score/review 컬럼이 없어 워드클라우드를 만들 수 없습니다.')

            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.subheader("부정(negative) 워드클라우드")
            st.caption("score <= 2 인 review 텍스트로 생성합니다.")
            if has_score and has_text:
                neg = df_work[pd.to_numeric(df_work["score"], errors="coerce") <= 2]
                neg_text = " ".join(neg["review"].fillna("").astype(str).tolist())
                if neg_text.strip():
                    fig = make_wordcloud(neg_text, is_dark_mode=is_dark)
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                else:
                    st.info("부정 리뷰 텍스트가 없어 워드클라우드를 만들 수 없습니다.")
            else:
                st.info('score/review 컬럼이 없어 워드클라우드를 만들 수 없습니다.')

            # =====================================
            # 키워드 분석
            # =====================================
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.header("3) 키워드/보이스 오브 커스터머")
            st.caption("리뷰 텍스트에서 자주 등장하는 단어를 Counter로 집계해 TOP 20을 보여줍니다.")

            kw_col1, kw_col2 = st.columns([1.2, 1])
            with kw_col1:
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">키워드 TOP 20</div>
                      <div class="card-desc">review 컬럼의 텍스트를 합쳐 단어 빈도를 계산합니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                top20 = []
                if has_text:
                    all_words = []
                    for t in df_work["review"].fillna("").astype(str).tolist():
                        all_words.extend(simple_tokenize(t))
                    counter = Counter(all_words)
                    top20 = counter.most_common(20)
                    if len(top20) == 0:
                        st.info("키워드를 추출할 리뷰 텍스트가 없습니다.")
                    else:
                        kw_df = pd.DataFrame(top20, columns=["단어", "빈도"])
                        st.dataframe(kw_df, use_container_width=True, height=420)
                else:
                    st.info('review 컬럼("review")이 없어 키워드 분석을 건너뜁니다.')

            with kw_col2:
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">요약</div>
                      <div class="card-desc">상위 키워드로 고객 목소리의 핵심을 빠르게 파악합니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                top_words = [w for w, _ in top20[:5]] if has_text and len(top20) > 0 else []
                if len(top_words) > 0:
                    st.markdown(
                        f"""
                        <div class="insight-box">
                          <div><b>TOP 5 키워드</b></div>
                          <div class="muted" style="margin-top:6px;">{", ".join(top_words)}</div>
                          <div class="muted" style="margin-top:10px;">이 키워드들이 반복적으로 언급되는지, 최근 기간에 집중되는지 확인해보세요.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="insight-box">
                          <div><b>TOP 5 키워드</b></div>
                          <div class="muted" style="margin-top:6px;">데이터가 없어 키워드를 계산할 수 없습니다.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ------ AI 해석 (키워드 기반 보이스 오브 커스터머) ------
                ai_text_kw = "키워드 데이터가 없어 AI 해석을 생성할 수 없습니다."
                if st.checkbox("AI 관점에서 키워드 해석 보기", value=True):
                    if has_text and len(top20) > 0:
                        keyword_summary = ", ".join([f"{w}({c})" for w, c in top20[:10]])
                        ai_prompt_kw = (
                            "다음은 서비스 리뷰 텍스트에서 추출한 상위 키워드와 등장 빈도입니다.\n"
                            f"{keyword_summary}\n\n"
                            "이 키워드를 기반으로 고객들이 어떤 경험을 하고 있고, 주요 요구/불만/칭찬 포인트가 무엇인지 5줄 이내로 요약해 주세요."
                        )
                        ai_text_kw = ask_openai(ai_prompt_kw)

                st.markdown(
                    f"""
                    <div class="insight-box" style="margin-top:10px;">
                      <div><b>AI 해석 (보이스 오브 커스터머)</b></div>
                      <div class="muted" style="margin-top:6px; white-space:pre-wrap;">{ai_text_kw}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # =====================================
            # 감성 분석
            # =====================================
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.header("4) 감성 분석")
            st.caption("평점 기준으로 감성 라벨(positive/neutral/negative)을 만들고 분포를 시각화합니다.")

            with st.container():
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">감성 분포</div>
                      <div class="card-desc">score 기준: 4 이상 positive, 3 neutral, 2 이하 negative</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                dist = None
                if has_score:
                    def score_to_label(x):
                        try:
                            v = float(x)
                        except Exception:
                            return None
                        if v >= 4:
                            return "positive"
                        if v == 3:
                            return "neutral"
                        return "negative"

                    df_work["감성"] = df_work["score"].apply(score_to_label)
                    dist = df_work["감성"].dropna().value_counts()
                    order = ["positive", "neutral", "negative"]
                    dist = dist.reindex(order).fillna(0).astype(int)
                    dist_df = dist.reset_index()
                    dist_df.columns = ["감성", "개수"]
                    dist_df = dist_df.set_index("감성")
                    st.bar_chart(dist_df["개수"], use_container_width=True)
                else:
                    st.info('score 컬럼("score")이 없어 감성 분석을 건너뜁니다.')

            # ------ AI 해석 (감성 분포 설명) ------
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">AI 해석 - 감성 결과</div>
                      <div class="card-desc">감성 분포를 기반으로 전체 만족도와 리스크를 요약합니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if dist is not None:
                    total_senti = int(dist.sum())
                    pos_n = int(dist.get("positive", 0))
                    neu_n = int(dist.get("neutral", 0))
                    neg_n = int(dist.get("negative", 0))
                    ai_prompt_senti = (
                        "서비스 리뷰의 감성 분석 결과입니다.\n\n"
                        f"- positive: {pos_n}건\n"
                        f"- neutral: {neu_n}건\n"
                        f"- negative: {neg_n}건\n\n"
                        "위 분포를 기준으로 전반적인 고객 만족도 수준과 리스크 요인을 5줄 이내로 요약해 주세요."
                    )
                    ai_text_senti = ask_openai(ai_prompt_senti)
                    st.markdown(
                        f"""
                        <div class="insight-box">
                          <div><b>AI 해석 (감성 요약)</b></div>
                          <div class="muted" style="margin-top:6px; white-space:pre-wrap;">{ai_text_senti}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="insight-box">
                          <div><b>AI 해석 (감성 요약)</b></div>
                          <div class="muted" style="margin-top:6px; white-space:pre-wrap;">
                            감성 분포를 계산할 데이터가 없어 AI 해석을 생성할 수 없습니다.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # =====================================
            # 인사이트
            # =====================================
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.header("5) 인사이트")
            st.caption("KPI/키워드/감성 결과를 바탕으로 빠르게 다음 액션을 정리합니다.")

            insight_left, insight_right = st.columns([1.2, 1])
            with insight_left:
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">핵심 요약</div>
                      <div class="card-desc">자동으로 생성되는 간단한 리포트 요약입니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                bullets = []
                bullets.append(f"전체 리뷰 수: {total_reviews:,}건")
                if avg_score is not None:
                    bullets.append(f"평균 평점: {avg_score:.2f}점")
                if has_date:
                    bullets.append(f"작성 기간: {period_text}")
                if has_score and "dist" in locals():
                    total_s = int(dist.sum())
                    if total_s > 0:
                        pos = int(dist.get("positive", 0))
                        neg = int(dist.get("negative", 0))
                        bullets.append(f"긍/부정 비율: positive {pos/total_s:.0%}, negative {neg/total_s:.0%}")
                if has_text and "counter" in locals() and len(counter) > 0:
                    bullets.append(f"TOP 키워드: {', '.join([w for w, _ in counter.most_common(3)])}")

                st.markdown(
                    "<div class='insight-box'>" + "<br/>".join([f"• {b}" for b in bullets]) + "</div>",
                    unsafe_allow_html=True,
                )

            with insight_right:
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">추천 액션</div>
                      <div class="card-desc">초보자도 바로 실행할 수 있는 체크리스트입니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                actions = [
                    "부정(negative) 리뷰 상위 키워드를 확인하고 개선 항목을 3개로 줄여보세요.",
                    "최근 2주에 리뷰가 급증/급감했는지(추이 차트) 확인해 원인을 기록하세요.",
                    "평점 1~2점 리뷰를 필터링해 '반복되는 불만'을 5개만 추려보세요.",
                ]
                st.markdown(
                    "<div class='insight-box'>" + "<br/>".join([f"• {a}" for a in actions]) + "</div>",
                    unsafe_allow_html=True,
                )

            # ------ AI 해석 (종합 인사이트) ------
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    """
                    <div class="card">
                      <div class="card-title">AI 종합 인사이트</div>
                      <div class="card-desc">지표/키워드/감성 결과를 한 번에 요약한 간단한 리포트입니다.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 간단한 텍스트 컨텍스트 구성
                context_lines = []
                context_lines.append(f"- 전체 리뷰 수: {total_reviews}")
                if avg_score is not None:
                    context_lines.append(f"- 평균 평점: {avg_score:.2f}")
                if has_date:
                    context_lines.append(f"- 작성 기간: {period_text}")
                if has_text and 'top20' in locals() and len(top20) > 0:
                    context_lines.append(
                        "- 상위 키워드: " + ", ".join([w for w, _ in top20[:8]])
                    )
                if has_score and 'dist' in locals() and dist is not None:
                    total_senti = int(dist.sum())
                    pos_n = int(dist.get("positive", 0))
                    neu_n = int(dist.get("neutral", 0))
                    neg_n = int(dist.get("negative", 0))
                    context_lines.append(
                        f"- 감성 분포: positive {pos_n}, neutral {neu_n}, negative {neg_n} (총 {total_senti})"
                    )

                ai_prompt_overall = (
                    "다음은 어떤 서비스의 리뷰 데이터를 요약한 분석 결과입니다.\n\n"
                    + "\n".join(context_lines)
                    + "\n\n"
                    "위 정보를 바탕으로 1) 현재 고객 경험 상태, 2) 즉시 개선해야 할 우선순위 2~3가지, 3) 유지/강화해야 할 강점을 간단한 bullet 형식으로 정리해 주세요."
                )
                ai_text_overall = ask_openai(ai_prompt_overall)

                st.markdown(
                    f"""
                    <div class="insight-box">
                      <div><b>AI 요약 리포트</b></div>
                      <div class="muted" style="margin-top:6px; white-space:pre-wrap;">{ai_text_overall}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("</div>", unsafe_allow_html=True)

