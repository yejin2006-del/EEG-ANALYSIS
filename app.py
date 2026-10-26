from __future__ import annotations

from dataclasses import fields
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Config, DEFAULT_BANDS, EPOCH_SECONDS, VERSION, analyze, export_zip, list_csv_members, synthetic_csv

st.set_page_config(page_title="EEG 분석실", page_icon="〰️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp { background: #f6f8fa; }
[data-testid="stSidebar"] { background: #ffffff; }
.block-container { padding-top: 2.3rem; max-width: 1480px; }
h1, h2, h3 { letter-spacing: -0.035em; }
[data-testid="stMetric"] { background: white; border: 1px solid #e1e8ed;
    padding: 18px 20px; border-radius: 14px; }
[data-testid="stMetricValue"] { color: #0a6b65; }
.eyebrow { color: #087e77; font-size: .77rem; letter-spacing: .16em; font-weight: 700; }
.intro { color: #5c6c7b; font-size: 1.04rem; margin-bottom: 1.5rem; }
.empty { padding: 36px; border: 1px dashed #ccd9df; border-radius: 16px;
    background: white; color: #536573; margin-top: 16px; }
</style>""", unsafe_allow_html=True)

defaults = Config().to_dict()
for key, value in defaults.items():
    if key != "bands": st.session_state.setdefault(key, value)
st.session_state.setdefault("bands_value", pd.DataFrame(DEFAULT_BANDS))
st.session_state.setdefault("input_mode", "내 ZIP·CSV")


def reset_config():
    for key, value in Config().to_dict().items():
        if key != "bands": st.session_state[key] = value
    st.session_state.bands_value = pd.DataFrame(DEFAULT_BANDS)
    st.session_state.pop("band_editor", None)


def choose_demo():
    st.session_state.input_mode = "합성 예제"
    st.session_state.first_minutes = 3.0


def apply_import():
    try:
        payload = json.loads(st.session_state.config_upload.getvalue())
        allowed = {f.name for f in fields(Config)}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError("analysis_config.json 형식이 아닙니다.")
        c = Config(**payload)
        c.validate()
        for key, value in c.to_dict().items():
            if key != "bands": st.session_state[key] = value
        st.session_state.bands_value = pd.DataFrame(c.bands)
        st.session_state.pop("band_editor", None)
        st.session_state.import_message = (True, "설정을 불러왔습니다. 분석 실행을 눌러 적용해 주세요.")
    except Exception as exc:
        st.session_state.import_message = (False, f"설정을 불러오지 못했습니다: {exc}")


with st.sidebar:
    st.markdown("<div class='eyebrow'>ANALYSIS SETTINGS</div>", unsafe_allow_html=True)
    st.title("분석 기준")
    st.caption("기준을 바꾼 뒤 ‘분석 실행’을 누르면 적용돼요.")
    with st.expander("저장된 설정 불러오기"):
        st.file_uploader("analysis_config.json", type=["json"], key="config_upload")
        st.button("설정 불러오기", on_click=apply_import, disabled=st.session_state.get("config_upload") is None)
        st.button("기본값으로 초기화", on_click=reset_config)
    if "import_message" in st.session_state:
        ok, message = st.session_state.import_message
        (st.success if ok else st.error)(message)
    st.subheader("01 · 분석 시간")
    st.number_input("기록 시작 후 첫 몇 분", min_value=0.5, max_value=1440.0,
                    step=0.5, key="first_minutes")
    st.caption("실제 입면 시각이 아닌, CSV의 첫 기록 시각을 기준으로 합니다.")

    st.subheader("02 · 잡음 제외")
    with st.expander("부착 불량", expanded=True):
        st.checkbox("부착 불량 제외", key="sensor_enabled")
        st.number_input("Sensor≠1 허용 비율 (%)", min_value=0.0, max_value=100.0,
                        step=0.5, key="sensor_bad_pct", disabled=not st.session_state.sensor_enabled)
    with st.expander("값 고정·신호 소실"):
        st.checkbox("낮은 표준편차 제외", key="low_sd_enabled")
        st.number_input("최소 표준편차 (원시 단위)", min_value=0.0, step=0.1,
                        key="min_sd", disabled=not st.session_state.low_sd_enabled)
        st.checkbox("연속 0 제외", key="zero_enabled")
        st.number_input("0 지속 시간 (초)", min_value=0.004, step=0.1,
                        key="zero_seconds", disabled=not st.session_state.zero_enabled)
        st.checkbox("같은 값 지속 제외", key="flat_enabled")
        st.number_input("같은 값 지속 시간 (초)", min_value=0.004, step=0.1,
                        key="flat_seconds", disabled=not st.session_state.flat_enabled)
    with st.expander("급변·포화·움직임"):
        st.checkbox("급격한 변화 제외", key="jump_enabled")
        st.number_input("인접 표본 변화량 문턱 (원시 단위)", min_value=0.0, step=10.0,
                        key="jump_threshold", disabled=not st.session_state.jump_enabled)
        st.number_input("급변 허용 비율 (%)", min_value=0.0, max_value=100.0, step=0.5,
                        key="jump_bad_pct", disabled=not st.session_state.jump_enabled)
        st.checkbox("포화 의심 평탄부 제외", key="saturation_enabled")
        st.number_input("포화 절댓값 문턱 (원시 단위)", min_value=0.0, step=10.0,
                        key="saturation_level", disabled=not st.session_state.saturation_enabled)
        st.number_input("포화 평탄부 지속 시간 (초)", min_value=0.004, step=0.004,
                        format="%.3f", key="saturation_seconds", disabled=not st.session_state.saturation_enabled)
        st.checkbox("움직임 제외", key="motion_enabled")
        st.number_input("3축 차분 RMS 문턱 (원시 단위)", min_value=0.0, step=0.1,
                        format="%.6f", key="motion_threshold", disabled=not st.session_state.motion_enabled)
        st.caption("움직임 기본 문턱은 첨부 보고서의 탐색값입니다. 장비 공통 기준이 아닙니다.")
    st.caption("숫자·시각 오류는 항상 제외합니다. 품질 문턱값은 연구용 설정이며 검증된 임상 기준이 아닙니다.")

    st.subheader("03 · 주파수 계산")
    st.selectbox("계산 방식", ["welch", "periodogram"], key="method",
                 format_func=lambda x: "Welch · 작은 구간을 평균" if x == "welch" else "Periodogram · 30초 전체 FFT")
    st.selectbox("창 함수", ["hann", "hamming", "boxcar"], key="window",
                 format_func=lambda x: {"hann": "Hann", "hamming": "Hamming", "boxcar": "직사각형"}[x])
    st.number_input("Welch 계산 구간 (초)", min_value=0.5, max_value=30.0, step=0.5,
                    key="segment_seconds", disabled=st.session_state.method != "welch")
    st.slider("Welch 겹침 (%)", 0, 90, step=5, key="overlap_pct", disabled=st.session_state.method != "welch")
    st.selectbox("Welch 스펙트럼 평균", ["mean", "median"], key="average",
                 format_func=lambda x: "평균" if x == "mean" else "중앙값",
                 disabled=st.session_state.method != "welch")
    st.caption("30초마다 결과를 계산합니다. 추가 필터 없이, 주파수 계산 구간의 평균만 제거합니다.")

    st.subheader("04 · 주파수 대역")
    st.caption("아래 표를 클릭해 이름·하한·상한을 수정하거나 행을 추가하세요.")
    edited_bands = st.data_editor(st.session_state.bands_value, key="band_editor", num_rows="dynamic",
                                  hide_index=True, width="stretch",
                                  column_config={"name": st.column_config.TextColumn("이름", required=True),
                                                 "low": st.column_config.NumberColumn("하한 Hz", min_value=0.0, step=0.5, required=True),
                                                 "high": st.column_config.NumberColumn("상한 Hz", min_value=0.0, step=0.5, required=True)})
    c1, c2 = st.columns(2)
    c1.number_input("전체 하한 Hz", min_value=0.0, step=0.5, key="total_low")
    c2.number_input("전체 상한 Hz", min_value=0.5, step=0.5, key="total_high")
    st.caption("상대 파워 = 각 대역 파워 ÷ 전체 대역 파워 × 100. 대역이 겹치면 합은 100%가 아닐 수 있어요.")
    with st.expander("장비 설정"):
        st.number_input("표본화율 Hz", min_value=10.0, max_value=5000.0, step=1.0, key="fs")
        st.caption("첨부 장비의 250Hz는 잠정값입니다. Time 열의 반복 시각만으로 표본화율을 정하지 않습니다.")

current = {k: st.session_state[k] for k in defaults if k != "bands"}
current["bands"] = edited_bands.to_dict("records")
config_error = None
try:
    config = Config(**current)
    config.validate()
except (ValueError, TypeError, KeyError) as exc:
    config_error = str(exc)

st.markdown("<div class='eyebrow'>CAPSTONE · EEG WORKBENCH</div>", unsafe_allow_html=True)
st.title("뇌파 분석실")
st.markdown("<div class='intro'>분석 기준을 직접 정하고, 두 채널의 파워와 제외 구간을 함께 확인하세요.</div>", unsafe_allow_html=True)
with st.container(border=True):
    left, right = st.columns([3, 1])
    with left:
        mode = st.radio("데이터 선택", ["내 ZIP·CSV", "합성 예제"], key="input_mode", horizontal=True)
    with right:
        st.button("3분 예제로 시작", on_click=choose_demo, width="stretch")
    uploaded = None
    selected_member = None
    source_error = None
    source = None
    filename = ""
    if mode == "합성 예제":
        @st.cache_data(show_spinner=False)
        def demo_bytes():
            return synthetic_csv()
        source, filename = BytesIO(demo_bytes()), "synthetic_demo_3min.csv"
        st.info("합성 예제 · 실제 참가자 데이터가 아닙니다. 30~60초의 CH2 값 고정, 120~150초의 부착 불량을 포함합니다.")
        source_id = (filename, len(source.getbuffer()))
    else:
        uploaded = st.file_uploader("ZIP 또는 CSV 파일", type=["zip", "csv"], key="eeg_upload",
                                    help="ZIP은 그대로 넣을 수 있습니다. CSV를 Excel에서 다시 저장할 필요가 없습니다.")
        source_id = None
        if uploaded is not None:
            source, filename = uploaded, uploaded.name
            source_id = (filename, uploaded.size, getattr(uploaded, "file_id", ""))
            try:
                members = list_csv_members(source, filename)
                if members:
                    selected_member = st.selectbox("ZIP 안에서 분석할 CSV", members)
                st.caption(f"{filename} · {uploaded.size / 1024**2:.1f} MiB · 원본을 수정하지 않습니다.")
            except Exception as exc:
                source_error = f"파일을 열 수 없습니다: {exc}"
                st.error(source_error)
    with st.expander("기록 정보 · 결과 파일에 함께 저장"):
        a, b, c = st.columns(3)
        a.text_input("참가자 ID", key="participant", placeholder="예: HYC001")
        b.text_input("실험일", key="protocol_day", placeholder="예: D9")
        c.selectbox("조건", ["미지정", "기기사용", "기기차단", "기준선", "기타"], key="condition")
        st.text_input("측정 날짜·메모", key="record_note", placeholder="파일명만으로 날짜·참가자를 자동 확정하지 않습니다.")

if config_error: st.error(f"분석 기준을 확인해 주세요: {config_error}")
run_col, note_col = st.columns([1, 3])
run = run_col.button("분석 실행", type="primary", width="stretch",
                     disabled=source is None or source_error is not None or config_error is not None, key="run_analysis")
note_col.caption(f"기록 시작 후 첫 {current['first_minutes']:g}분 · 30초 단위 · CH1/CH2 개별 품질 검사")

def config_signature(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)

if run:
    bar = st.progress(0, text="분석을 준비하고 있습니다")
    try:
        with st.spinner("원본을 나누어 읽고 분석 중입니다. 설정 변경 시 자동으로 재실행되지 않습니다."):
            result = analyze(source, filename, config, selected_member,
                             progress=lambda fraction, message: bar.progress(fraction, text=message))
        st.session_state.result = result
        st.session_state.result_signature = config_signature(current)
        st.session_state.result_source = (source_id, selected_member)
        # Labels belong to the run; later text edits must not silently relabel an old result.
        st.session_state.result_labels = {k: st.session_state.get(k, "")
                                          for k in ["participant", "protocol_day", "condition", "record_note"]}
    except Exception as exc:
        st.error(f"분석을 완료하지 못했습니다: {exc}")
        st.caption("기존 결과가 있으면 아래에 유지됩니다. 기존 결과의 파일·설정을 확인해 주세요.")
    finally:
        bar.empty()

if "result" not in st.session_state:
    st.markdown("""<div class='empty'><b>파일을 선택하고 분석 실행을 눌러 주세요.</b><br/><br/>
    왼쪽에서 ① 첫 몇 분 ② 잡음 제외 ③ 계산 방식 ④ 주파수 대역을 정할 수 있어요.<br/>
    ‘3분 예제로 시작’을 누르면 실제 파일 없이 화면과 제외 규칙을 확인할 수 있습니다.</div>""", unsafe_allow_html=True)
    st.stop()

r = st.session_state.result
m = r["metadata"]
if st.session_state.result_signature != config_signature(current) or st.session_state.result_source != (source_id, selected_member):
    st.warning("현재 입력이 바뀌었습니다. 아래는 이전 실행 결과입니다. 새 기준을 적용하려면 분석 실행을 눌러 주세요.")
st.caption(f"표시 중인 실행: {m['source_filename']} {(' / ' + m['archive_member']) if m['archive_member'] else ''} · 첫 {m['requested_minutes']:g}분 · {r['config']['method']} · {m['created_utc']}")

lead = r["summary"].iloc[0]
mc = st.columns(4)
mc[0].metric(f"{lead['band']} 중앙값", "산출 불가" if pd.isna(lead.median_pct) else f"{lead.median_pct:.2f}%")
mc[1].metric("유효 시간", f"{m['valid_minutes']:.1f}분")
mc[2].metric("요청 시간 대비 유효 비율", f"{m['valid_pct_of_requested']:.1f}%")
mc[3].metric("유효 채널·구간", f"{int(r['epochs'].valid.sum()):,} / {len(r['epochs']):,}")
if not m["eligible_for_night_comparison"]:
    st.warning("밤 비교 포함 기준 미충족: 요청 시간의 50% 이상이면서 최소 20분의 유효 데이터가 필요합니다. 표시된 값은 확보 구간의 기술 통계입니다.")
if m["selected_elapsed_minutes"] < m["requested_minutes"] - 0.1:
    st.info(f"파일의 확보 범위는 약 {m['selected_elapsed_minutes']:.1f}분입니다. 요청한 시간을 채우지 못한 부분은 보간하지 않았습니다.")
rate = m["observed_rows_per_wall_second"]
if rate and abs(rate - r["config"]["fs"]) / r["config"]["fs"] > 0.05:
    st.warning(f"행 수/경과시간은 약 {rate:.2f}행/초로, 설정한 {r['config']['fs']:g}Hz와 차이가 있습니다. 기록 공백과 장비 표본화율을 확인해 주세요.")

tab1, tab2, tab3, tab4 = st.tabs(["파워 추세", "잡음·제외 내역", "파형·스펙트럼", "결과 다운로드"])


def style_chart(fig, x_title, y_title, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=16, r=16, t=24, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", xaxis_title=x_title,
                      yaxis_title=y_title, legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


with tab1:
    st.subheader("시간에 따른 상대 파워")
    st.caption("유효 채널의 비율을 30초마다 평균합니다. 제외 구간은 빈 값으로 유지합니다.")
    band_names = r["summary"].band.tolist()
    shown = st.multiselect("표시할 대역", band_names, default=band_names[:1], key="shown_bands")
    timeline = r["timeline"]
    fig = go.Figure()
    colors = ["#087e77", "#ec9b32", "#596bcc", "#c05e95", "#71933b"]
    # NaN separators preserve both excluded epochs and long time gaps.
    x = timeline.start_seconds.to_numpy() / 60
    for i, band in enumerate(shown):
        y = timeline[f"relative_pct_{band}"].to_numpy()
        xx, yy = [], []
        for j in range(len(x)):
            if j and (not np.isfinite(x[j - 1]) or not np.isfinite(x[j]) or x[j] - x[j - 1] > 0.6 or x[j] < x[j - 1]):
                xx.append(None); yy.append(None)
            xx.append(float(x[j]) if np.isfinite(x[j]) else None)
            yy.append(float(y[j]) if np.isfinite(y[j]) else None)
        fig.add_trace(go.Scatter(x=xx, y=yy, name=band, mode="lines+markers", connectgaps=False,
                                 line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=4)))
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(style_chart(fig, "기록 시작 후 시간 (분)", "상대 파워 (%)"), width="stretch")
    st.subheader("대역별 요약")
    st.dataframe(r["summary"].rename(columns={"band": "대역", "low_hz": "하한 Hz", "high_hz": "상한 Hz",
                   "median_pct": "중앙값 %", "mean_pct": "평균 %", "valid_epochs": "유효 구간 수"}),
                   hide_index=True, width="stretch")
    st.caption("상대 파워는 주파수 대역의 에너지 비율입니다. 깊은 수면 시간의 비율이나 수면 단계로 해석하지 않습니다.")

with tab2:
    st.subheader("채널별 품질 판정")
    ep = r["epochs"]
    quality = ep.groupby("channel").valid.agg(["sum", "count"]).reset_index()
    quality["excluded"] = quality["count"] - quality["sum"]
    st.dataframe(quality.rename(columns={"channel": "채널", "sum": "통과 구간", "count": "전체 구간", "excluded": "제외 구간"}),
                 hide_index=True, width="stretch")
    excluded = ep.loc[~ep.valid]
    if excluded.empty:
        st.success("현재 설정에서 제외된 채널·구간이 없습니다.")
    else:
        reasons = excluded.reason.str.split(" / ").explode().value_counts().rename_axis("제외 사유").reset_index(name="채널·구간 수")
        st.dataframe(reasons, hide_index=True, width="stretch")
        st.caption("하나의 구간에 여러 사유가 있으면 각 사유에 중복 집계합니다.")
    only_excluded = st.checkbox("제외 구간만 보기", value=True)
    st.dataframe(excluded if only_excluded else ep, hide_index=True, width="stretch", height=350)
    st.caption(f"끝부분의 {m['trailing_samples_not_analyzed']:,}개 표본은 완전한 30초가 아니어서 분석하지 않았습니다. 전체 구간 표는 다운로드할 수 있습니다.")

with tab3:
    st.subheader("실제 파형 예시")
    options = list(r["preview"])
    example = st.selectbox("대표 구간", options, format_func=lambda k: (
        f"{r['preview'][k]['channel']} · {'통과' if k.endswith('_valid') else '제외'} · 구간 {r['preview'][k]['epoch_id']}"))
    p = r["preview"][example]
    st.caption(f"기록 시작 후 {p['start_seconds']:.1f}초부터 · {p['reason'] or '현재 품질 규칙 통과'}")
    waveform = go.Figure(go.Scattergl(x=p["time"], y=p["signal"], mode="lines", line=dict(color="#087e77", width=1)))
    st.plotly_chart(style_chart(waveform, "구간 내 표본 기준 시간 (초)", "진폭 (원시 단위)", 300), width="stretch")
    st.subheader("주파수 스펙트럼 예시")
    spectrum = go.Figure()
    for ch, ps in r["spectra"].items():
        mask = ps["frequencies"] <= r["config"]["total_high"]
        spectrum.add_trace(go.Scatter(x=ps["frequencies"][mask], y=ps["psd"][mask],
                          name=f"CH{ch+1} · 구간 {ps['epoch_id']}", mode="lines"))
    if not r["spectra"]:
        st.info("품질 기준을 통과한 구간이 없어 스펙트럼 예시가 없습니다.")
    else:
        spectrum.update_yaxes(type="log")
        st.plotly_chart(style_chart(spectrum, "주파수 (Hz)", "PSD (원시 단위²/Hz)", 320), width="stretch")
    st.caption("채널별 첫 통과 구간의 스펙트럼입니다. 두 채널의 예시 구간이 다를 수 있습니다. 전체 밤 평균 스펙트럼이 아닙니다.")

with tab4:
    st.subheader("결과와 적용 기준을 함께 저장")
    st.download_button("분석 결과 전체 ZIP 다운로드", data=export_zip(r, st.session_state.result_labels),
                       file_name="eeg_analysis_results.zip", mime="application/zip", type="primary")
    c1, c2 = st.columns(2)
    c1.download_button("적용한 설정 JSON", data=json.dumps(r["config"], ensure_ascii=False, indent=2),
                        file_name="analysis_config.json", mime="application/json")
    c2.download_button("채널별 결과 CSV", data=r["epochs"].to_csv(index=False).encode("utf-8-sig"),
                        file_name="epoch_channel_results.csv", mime="text/csv")
    st.caption("ZIP: 채널별 결과 · 채널 통합 결과 · 대역별 요약 · 분석 설정 · 실행 정보 · 기록 정보")
    with st.expander("이 실행에 적용한 설정"):
        st.json(r["config"])
    with st.expander("파일·시간·재현 정보"):
        st.json(m)
        st.json(st.session_state.result_labels)
    st.caption("빈 값은 0으로 바꾸지 않습니다. 원본 전체의 SHA256을 저장하지만 품질 검사는 요청한 첫 시간창에만 적용합니다.")

st.divider()
st.caption(f"EEG 분석실 v{VERSION} · 연구용 상대 파워 분석 · 수면 단계·손상·회복시간 자동 판정은 포함하지 않습니다.")
