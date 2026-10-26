"""Streaming, configurable two-channel EEG power analysis (research use)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
from typing import BinaryIO, Callable
import zipfile

import numpy as np
import pandas as pd
from scipy import signal

VERSION = "1.0.1"
EPOCH_SECONDS = 30
DEFAULT_BANDS = [
    {"name": "Delta", "low": 0.5, "high": 4.0},
    {"name": "Theta", "low": 4.0, "high": 8.0},
    {"name": "Alpha", "low": 8.0, "high": 12.0},
    {"name": "Sigma", "low": 12.0, "high": 16.0},
    {"name": "Beta", "low": 16.0, "high": 30.0},
]
REASONS = {
    "NONFINITE": "EEG 값 누락·비정상 숫자", "TIME_INVALID": "시각 해석 실패",
    "TIME_REVERSE": "시각 역행", "TIME_GAP": "1초 초과 기록 공백",
    "TIME_DURATION": "30초 표본과 실제 경과 불일치", "SENSOR": "부착 불량",
    "LOW_SD": "신호 변동이 너무 작음", "ZERO_RUN": "연속 0",
    "FLATLINE": "같은 값이 지속됨", "JUMP": "급격한 변화",
    "SATURATION": "포화 의심 평탄부", "MOTION": "움직임 기준 초과",
    "MOTION_MISSING": "움직임 값 누락", "ZERO_POWER": "전체 파워 0·계산 불가",
}


@dataclass
class Config:
    first_minutes: float = 180.0
    fs: float = 250.0
    method: str = "welch"
    segment_seconds: float = 4.0
    overlap_pct: int = 50
    window: str = "hann"
    average: str = "mean"
    total_low: float = 0.5
    total_high: float = 30.0
    bands: list[dict] = field(default_factory=lambda: [dict(b) for b in DEFAULT_BANDS])
    sensor_enabled: bool = True
    sensor_bad_pct: float = 1.0
    low_sd_enabled: bool = True
    min_sd: float = 0.2
    zero_enabled: bool = True
    zero_seconds: float = 1.0
    flat_enabled: bool = True
    flat_seconds: float = 2.0
    jump_enabled: bool = True
    jump_threshold: float = 100.0
    jump_bad_pct: float = 1.0
    saturation_enabled: bool = True
    saturation_level: float = 390.0
    saturation_seconds: float = 0.02
    motion_enabled: bool = False
    motion_threshold: float = 0.767232

    def validate(self) -> None:
        numeric = [v for v in asdict(self).values() if isinstance(v, (float, int))]
        if not all(np.isfinite(v) for v in numeric):
            raise ValueError("설정에는 유한한 숫자만 입력해 주세요.")
        if not 0.5 <= self.first_minutes <= 1440 or not 10 <= self.fs <= 5000:
            raise ValueError("분석 시간은 0.5~1440분, 표본화율은 10~5000Hz여야 합니다.")
        if not (0 <= self.total_low < self.total_high <= self.fs / 2):
            raise ValueError("전체 파워 대역은 0 이상, 상한은 표본화율의 절반 이하여야 합니다.")
        if self.method not in {"welch", "periodogram"}:
            raise ValueError("지원하는 계산 방식은 welch / periodogram입니다.")
        if self.window not in {"hann", "hamming", "boxcar"} or self.average not in {"mean", "median"}:
            raise ValueError("계산 창 또는 평균 방식 설정을 확인해 주세요.")
        if not 0.5 <= self.segment_seconds <= EPOCH_SECONDS or not 0 <= self.overlap_pct <= 90:
            raise ValueError("Welch 구간은 0.5~30초, 겹침은 0~90%여야 합니다.")
        if not self.bands or len(self.bands) > 16:
            raise ValueError("주파수 대역은 1~16개를 입력해 주세요.")
        names = []
        for band in self.bands:
            name = str(band.get("name", "")).strip()
            low, high = float(band["low"]), float(band["high"])
            if not name or len(name) > 40 or name in names:
                raise ValueError("대역 이름은 비어 있지 않은, 서로 다른 40자 이하 이름으로 입력해 주세요.")
            if not (self.total_low <= low < high <= self.total_high):
                raise ValueError(f"{name}: 대역은 전체 파워 범위 안에 있어야 합니다.")
            names.append(name)
        for v in [self.sensor_bad_pct, self.jump_bad_pct]:
            if not 0 <= v <= 100:
                raise ValueError("제외 비율 문턱은 0~100%로 입력해 주세요.")
        for v in [self.min_sd, self.zero_seconds, self.flat_seconds, self.jump_threshold,
                  self.saturation_level, self.saturation_seconds, self.motion_threshold]:
            if v < 0:
                raise ValueError("잡음 문턱값에는 음수를 입력할 수 없습니다.")
        if min(self.zero_seconds, self.flat_seconds, self.saturation_seconds) <= 0:
            raise ValueError("연속 지속 시간 문턱은 0초보다 커야 합니다.")

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def binary_source(source: str | Path | BinaryIO):
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            yield f
    else:
        source.seek(0)
        try:
            yield source
        finally:
            source.seek(0)


def list_csv_members(source, filename: str) -> list[str]:
    if filename.lower().endswith(".csv"):
        return []
    if not filename.lower().endswith(".zip"):
        raise ValueError("CSV 또는 ZIP 파일을 선택해 주세요.")
    with binary_source(source) as f, zipfile.ZipFile(f) as z:
        members = [i.filename for i in z.infolist() if not i.is_dir()
                   and i.filename.lower().endswith(".csv") and "__MACOSX/" not in i.filename]
        if not members:
            raise ValueError("ZIP 안에 CSV 파일이 없습니다.")
        if len(set(members)) != len(members):
            raise ValueError("ZIP 내부에 같은 경로의 CSV가 중복되어 있습니다. 원본을 확인해 주세요.")
        return members


@contextmanager
def csv_stream(source, filename: str, member: str | None):
    with binary_source(source) as f:
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                if member is None:
                    raise ValueError("ZIP 안에서 분석할 CSV를 선택해 주세요.")
                with z.open(member) as raw, TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    yield text
        else:
            # detach keeps a Streamlit UploadedFile open after this context exits.
            text = TextIOWrapper(f, encoding="utf-8-sig", newline="")
            try:
                yield text
            finally:
                text.detach()


class ClockDecoder:
    """Unwrap midnight without deleting repeated timestamps or hiding reversals."""
    def __init__(self):
        self.previous = None
        self.day = 0

    def decode(self, values: pd.Series) -> np.ndarray:
        pieces = values.str.strip().str.extract(r"^(\d{2}):(\d{2}):(\d{2}(?:\.\d{1,9})?)$")
        a = pieces.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(a).all(axis=1) & (a[:, 0] < 24) & (a[:, 1] < 60) & (a[:, 2] < 60)
        seconds = a[:, 0] * 3600 + a[:, 1] * 60 + a[:, 2]
        out = np.full(len(values), np.nan)
        idx = np.flatnonzero(valid)
        if idx.size:
            s = seconds[idx]
            prev = s[0] if self.previous is None else self.previous
            midnight = np.diff(s, prepend=prev) < -43200
            day_offsets = self.day + np.cumsum(midnight)
            out[idx] = s + day_offsets * 86400
            self.day = int(day_offsets[-1])
            self.previous = float(s[-1])
        return out


def longest_run(mask: np.ndarray) -> int:
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def longest_constant(x: np.ndarray, minimum_abs: float = 0) -> int:
    cuts = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1, len(x)]
    lengths = np.diff(cuts)
    eligible = np.abs(x[cuts[:-1]]) >= minimum_abs
    return int(np.max(lengths[eligible])) if eligible.any() else 0


def compute_psd(x: np.ndarray, c: Config):
    win_length = int(round(c.segment_seconds * c.fs)) if c.method == "welch" else len(x)
    win = signal.get_window(c.window, win_length, fftbins=True)
    common = dict(fs=c.fs, window=win, detrend="constant", scaling="density", return_onesided=True)
    if c.method == "welch":
        return signal.welch(x, nperseg=win_length,
                            noverlap=int(win_length * c.overlap_pct / 100),
                            nfft=win_length, average=c.average, **common)
    return signal.periodogram(x, nfft=len(x), **common)


def band_power(f: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    # Interpolate exact boundaries so non-grid edits and shared endpoints integrate consistently.
    inner = (f > low) & (f < high)
    grid = np.r_[low, f[inner], high]
    y = np.r_[np.interp(low, f, psd), psd[inner], np.interp(high, f, psd)]
    return float(np.trapezoid(y, grid))


def epoch_quality(t: np.ndarray, data: np.ndarray, channel: int, c: Config) -> tuple[list[str], dict]:
    x, sensor, motion = data[:, channel], data[:, 2], data[:, 3:6]
    reasons = []
    info = {"sd_raw": float(np.std(x)), "sensor_bad_pct": float(np.mean(sensor != 1) * 100)}
    if not np.isfinite(x).all(): reasons.append("NONFINITE")
    if not np.isfinite(t).all():
        reasons.append("TIME_INVALID")
    else:
        dt = np.diff(t)
        if (dt < 0).any(): reasons.append("TIME_REVERSE")
        if (dt > 1).any(): reasons.append("TIME_GAP")
        span = t[-1] - t[0] + 1 / c.fs
        if not 0.9 * EPOCH_SECONDS <= span <= 1.1 * EPOCH_SECONDS:
            reasons.append("TIME_DURATION")
    if c.sensor_enabled and info["sensor_bad_pct"] > c.sensor_bad_pct: reasons.append("SENSOR")
    if np.isfinite(x).all():
        if c.low_sd_enabled and info["sd_raw"] < c.min_sd: reasons.append("LOW_SD")
        if c.zero_enabled and longest_run(x == 0) >= int(np.ceil(c.zero_seconds * c.fs)):
            reasons.append("ZERO_RUN")
        if c.flat_enabled and longest_constant(x) >= int(np.ceil(c.flat_seconds * c.fs)):
            reasons.append("FLATLINE")
        jumps = float(np.mean(np.abs(np.diff(x)) > c.jump_threshold) * 100)
        info["jump_bad_pct"] = jumps
        if c.jump_enabled and jumps > c.jump_bad_pct: reasons.append("JUMP")
        if c.saturation_enabled and longest_constant(x, c.saturation_level) >= int(np.ceil(c.saturation_seconds * c.fs)):
            reasons.append("SATURATION")
    if c.motion_enabled:
        if not np.isfinite(motion).all():
            reasons.append("MOTION_MISSING")
        else:
            rms = float(np.sqrt(np.mean(np.diff(motion, axis=0) ** 2)))
            info["motion_rms_raw"] = rms
            if rms > c.motion_threshold: reasons.append("MOTION")
    return reasons, info


def analyze(source, filename: str, config: Config, member: str | None = None,
            progress: Callable[[float, str], None] | None = None, chunk_rows: int = 100_000) -> dict:
    config.validate()
    c = config
    with binary_source(source) as raw:
        digest = sha256()
        for chunk in iter(lambda: raw.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    file_hash = digest.hexdigest()
    if progress: progress(0.02, "파일 확인 완료 · CSV를 나누어 읽고 있습니다")
    samples_per_epoch = int(round(c.fs * EPOCH_SECONDS))
    duration = c.first_minutes * 60
    records, combined, preview, spectra = [], [], {}, {}
    pending_t, pending_x = np.empty(0), np.empty((0, 6))
    decoder = ClockDecoder()
    origin = None
    processed = 0
    scanned = 0
    selected_rows = 0
    last_selected = None
    last_epoch_time = None
    elapsed_max = 0.0
    selection_ended = False
    columns = None
    missing_optional = []
    invalid_time_rows = 0
    with csv_stream(source, filename, member) as stream:
        reader = pd.read_csv(stream, dtype=str, chunksize=chunk_rows, on_bad_lines="error")
        for chunk in reader:
            scanned += len(chunk)
            chunk.columns = [str(col).strip().lower() for col in chunk.columns]
            if columns is None:
                columns = list(chunk.columns)
                if len(set(columns)) != len(columns): raise ValueError("중복된 열 이름이 있습니다.")
                missing = {"time", "ch1", "ch2"} - set(columns)
                if missing: raise ValueError("필수 열 누락: " + ", ".join(sorted(missing)))
                missing_optional = [col for col in ["sensor", "x", "y", "z"] if col not in columns]
                if c.sensor_enabled and "sensor" not in columns:
                    raise ValueError("sensor 열이 없습니다. 부착 불량 제외를 끄거나 원본을 확인해 주세요.")
                if c.motion_enabled and any(col not in columns for col in ["x", "y", "z"]):
                    raise ValueError("움직임 제외를 사용하려면 x, y, z 열이 모두 필요합니다.")
            t = decoder.decode(chunk["time"])
            if origin is None:
                if not len(t) or not np.isfinite(t[0]):
                    raise ValueError("첫 행의 시각을 해석할 수 없습니다. HH:MM:SS.mmm 형식을 확인해 주세요.")
                origin = float(t[0])
            stop_indices = np.flatnonzero(np.isfinite(t) & (t >= origin + duration))
            # Stop at the first boundary; do not concatenate later time reversals into the selected window.
            end = int(stop_indices[0]) if len(stop_indices) else len(chunk)
            if len(stop_indices): selection_ended = True
            chunk, t = chunk.iloc[:end], t[:end]
            selected_rows += len(t)
            invalid_time_rows += int(np.sum(~np.isfinite(t)))
            x = np.column_stack([
                pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=float)
                if col in chunk else np.full(len(chunk), np.nan)
                for col in ["ch1", "ch2", "sensor", "x", "y", "z"]
            ])
            if len(t):
                finite = t[np.isfinite(t)]
                if finite.size:
                    last_selected = float(finite[-1])
                    elapsed_max = max(elapsed_max, float(np.max(finite)) - origin + 1 / c.fs)
                pending_t = np.concatenate([pending_t, t])
                pending_x = np.concatenate([pending_x, x])
            offset = 0
            while len(pending_t) - offset >= samples_per_epoch:
                et = pending_t[offset:offset + samples_per_epoch]
                ex = pending_x[offset:offset + samples_per_epoch]
                offset += samples_per_epoch
                eid = processed
                processed += 1
                start_sec = float(et[0] - origin) if np.isfinite(et[0]) else float("nan")
                valid_channels = []
                boundary_reason = []
                if last_epoch_time is not None and np.isfinite(et[0]) and np.isfinite(last_epoch_time):
                    gap = et[0] - last_epoch_time
                    if gap < 0: boundary_reason.append("TIME_REVERSE")
                    if gap > 1: boundary_reason.append("TIME_GAP")
                last_epoch_time = et[-1]
                for ch in range(2):
                    reasons, metrics = epoch_quality(et, ex, ch, c)
                    reasons = list(dict.fromkeys(reasons + boundary_reason))
                    # Keep the first minute and representative pass/fail raw examples, never all raw samples.
                    row = {"epoch_id": eid, "channel": f"CH{ch + 1}", "start_seconds": start_sec,
                           "end_seconds": float(et[-1] - origin) if np.isfinite(et[-1]) else float("nan"),
                           "source_sample_start": eid * samples_per_epoch,
                           "source_sample_end_exclusive": (eid + 1) * samples_per_epoch, **metrics}
                    for b in c.bands:
                        row[f"power_{b['name']}"] = np.nan
                        row[f"relative_pct_{b['name']}"] = np.nan
                    row["total_power"] = np.nan
                    if not reasons:
                        f, p = compute_psd(ex[:, ch], c)
                        total = band_power(f, p, c.total_low, c.total_high)
                        if not np.isfinite(total) or total <= 0:
                            reasons.append("ZERO_POWER")
                        else:
                            row["total_power"] = total
                            for b in c.bands:
                                bp = band_power(f, p, float(b["low"]), float(b["high"]))
                                row[f"power_{b['name']}"] = bp
                                row[f"relative_pct_{b['name']}"] = 100 * bp / total
                            valid_channels.append(row)
                            if ch not in spectra:
                                spectra[ch] = {"frequencies": f, "psd": p, "epoch_id": eid,
                                               "start_seconds": start_sec}
                    row["valid"] = not reasons
                    row["reason_codes"] = "|".join(reasons)
                    row["reason"] = " / ".join(REASONS[r] for r in reasons)
                    records.append(row)
                    example_key = f"{ch}_{'valid' if not reasons else 'excluded'}"
                    if example_key not in preview:
                        preview[example_key] = {"time": np.arange(len(et)) / c.fs,
                                                "signal": ex[:, ch].copy(), "epoch_id": eid,
                                                "start_seconds": start_sec, "channel": f"CH{ch+1}",
                                                "reason": row["reason"]}
                cr = {"epoch_id": eid, "start_seconds": start_sec, "valid_channels": len(valid_channels)}
                clock_span = float(et[-1] - et[0] + 1 / c.fs) if np.isfinite(et[[0, -1]]).all() else 0
                cr["valid_seconds"] = max(0, clock_span) if valid_channels else 0.0
                for b in c.bands:
                    vals = [r[f"relative_pct_{b['name']}"] for r in valid_channels]
                    cr[f"relative_pct_{b['name']}"] = float(np.mean(vals)) if vals else np.nan
                combined.append(cr)
            pending_t, pending_x = pending_t[offset:], pending_x[offset:]
            if progress:
                progress(min(0.97, 0.02 + 0.95 * elapsed_max / duration),
                         f"{min(elapsed_max / 60, c.first_minutes):.1f} / {c.first_minutes:g}분 · {processed:,}개 구간 확인")
            if selection_ended: break
    if origin is None: raise ValueError("데이터 행이 없는 CSV입니다.")
    epochs, timeline = pd.DataFrame(records), pd.DataFrame(combined)
    if epochs.empty:
        raise ValueError("완전한 30초 분석 구간이 없습니다. 파일 길이·표본화율을 확인해 주세요.")
    valid_seconds = min(float(timeline.valid_seconds.sum()), duration)
    coverage_pct = valid_seconds / duration * 100
    eligible = valid_seconds >= max(20 * 60, duration * 0.5)
    summaries = []
    for b in c.bands:
        v = timeline[f"relative_pct_{b['name']}"]
        summaries.append({"band": b["name"], "low_hz": float(b["low"]), "high_hz": float(b["high"]),
                          "median_pct": float(v.median()) if v.notna().any() else None,
                          "mean_pct": float(v.mean()) if v.notna().any() else None,
                          "valid_epochs": int(v.notna().sum())})
    metadata = {
        "program_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": filename, "archive_member": member, "source_sha256": file_hash,
        "config_sha256": sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest(),
        "anchor": "recording_start", "recording_start_clock_seconds": origin,
        "requested_minutes": c.first_minutes, "selected_elapsed_minutes": elapsed_max / 60,
        "selected_sample_count": selected_rows, "rows_parsed_in_chunks": scanned,
        "complete_epochs": processed, "trailing_samples_not_analyzed": len(pending_t),
        "invalid_time_rows": invalid_time_rows, "valid_minutes": valid_seconds / 60,
        "valid_pct_of_requested": coverage_pct, "eligible_for_night_comparison": eligible,
        "comparison_rule": "valid >= max(20 minutes, 50% of requested window)",
        "observed_rows_per_wall_second": selected_rows / elapsed_max if elapsed_max > 0 else None,
        "source_read_scope": "requested window plus parser chunk; not full file QC",
        "stopped_at_window": selection_ended, "missing_optional_columns": missing_optional,
        "epoch_seconds": EPOCH_SECONDS, "power_unit": "raw_unit_squared",
        "fs_status": "user supplied; 250 Hz is provisional for the attached device",
        "filters": "no additional filter; constant detrend within PSD segments",
        "channel_combine": "mean of relative powers of valid channels per epoch",
        "valid_time_definition": "sum of accepted epoch wall spans; gaps and rejected epochs excluded",
        "summary_scope": "descriptive; use eligibility flag before comparing nights",
    }
    if progress: progress(1.0, "분석 완료")
    return {"epochs": epochs, "timeline": timeline, "summary": pd.DataFrame(summaries),
            "metadata": metadata, "config": c.to_dict(), "preview": preview, "spectra": spectra}


def export_zip(result: dict, annotations: dict | None = None) -> bytes:
    out = BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for key, filename in [("epochs", "epoch_channel_results.csv"),
                              ("timeline", "epoch_combined_results.csv"), ("summary", "band_summary.csv")]:
            z.writestr(filename, result[key].to_csv(index=False).encode("utf-8-sig"))
        payloads = {"analysis_config.json": result["config"], "run_metadata.json": result["metadata"],
                    "record_labels.json": annotations or {}}
        for name, value in payloads.items():
            z.writestr(name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        z.writestr("READ_ME.txt", "Relative power is a PSD integral ratio, not sleep-stage time.\n"
                   "Excluded values remain missing, not zero. No sleep stage or recovery is inferred.\n"
                   "Settings, raw units, provisional fs, eligibility and source hash are in the JSON files.\n"
                   "Only the requested beginning of the selected CSV was inspected.\n")
    return out.getvalue()


def synthetic_csv(fs: int = 250, seconds: int = 180) -> bytes:
    """Explicitly synthetic demo: includes one flat CH2 epoch and one detached epoch."""
    rng = np.random.default_rng(42)
    t = np.arange(fs * seconds) / fs
    ch1 = 24 * np.sin(2 * np.pi * 2 * t) + 12 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 2, len(t))
    ch2 = 20 * np.sin(2 * np.pi * 2 * t + 0.2) + 14 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 2, len(t))
    ch2[(t >= 30) & (t < 60)] = 0
    sensor = np.ones(len(t), dtype=int)
    sensor[(t >= 120) & (t < 150)] = 0
    clock = pd.to_timedelta(t + 3600, unit="s")
    milliseconds = np.round(clock.total_seconds().to_numpy() * 1000).astype(int)
    times = [f"{v//3600000%24:02d}:{v//60000%60:02d}:{v//1000%60:02d}.{v%1000:03d}" for v in milliseconds]
    df = pd.DataFrame({"time": times, "ch1": ch1, "ch2": ch2, "sensor": sensor,
                       "x": 115, "y": 190, "z": 130, "battery": 98, "beat": "None"})
    return df.to_csv(index=False, float_format="%.6f").encode("utf-8")
