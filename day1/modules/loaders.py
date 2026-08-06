"""실데이터 로더 + 합성 폴백 통합 인터페이스 — Day 1 실습 공용.

원칙
  1) 사전 배포된 경량 서브셋(GitHub raw)에서 다운로드를 시도한다.
  2) 실패하면 mfg_datagen 의 동일 스키마 합성 데이터로 자동 대체한다.
  3) 두 경로의 반환 스키마(컬럼명·센서 개수·라벨 형식·shape 규약)는 완전히 동일하다.
     이후 셀은 데이터가 어느 경로로 왔는지 몰라도 동작한다.

강사 사전 준비: DATA_BASE 를 본인 GitHub 저장소 raw 주소로 바꾸고
DAY1_운영가이드.md 의 절차대로 서브셋·체크포인트를 업로드한다.
환경변수 MFG_FORCE_FALLBACK=1 이면 네트워크를 시도하지 않고 바로 폴백한다(리허설용).
"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

try:
    from . import mfg_datagen  # 패키지로 임포트된 경우
except ImportError:
    import mfg_datagen         # Colab에서 파일 단독 다운로드된 경우

# ▼▼ 강사가 강의 전에 본인 저장소 주소로 교체 ▼▼
DATA_BASE = os.environ.get(
    "MFG_DATA_BASE",
    "https://raw.githubusercontent.com/leejiyoon52/ai-course/main/day1",
)

FALLBACK_MSG = "⚠️ 실데이터 수신 실패 — 합성 데이터로 진행합니다"
CKPT_MSG = "체크포인트 미수신 — 데모 모델로 진행합니다(품질 저하 가능)"

_CMAPSS_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def _fetch(rel_path, timeout=15):
    """DATA_BASE/rel_path 를 받아 bytes 반환. 실패 시 None (예외를 밖으로 내지 않는다)."""
    if os.environ.get("MFG_FORCE_FALLBACK") == "1":
        return None
    url = f"{DATA_BASE}/{rel_path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


# ================================================================ C-MAPSS
def _parse_cmapss_txt(raw):
    """NASA 원본 txt(공백 구분 26열)를 DataFrame으로."""
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = _CMAPSS_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    # s17·s18 등이 정수로 읽히는 경우가 있어 합성 폴백과 dtype을 통일한다
    df[df.columns[2:]] = df[df.columns[2:]].astype("float64")
    return df


def _synth_cmapss(subset, seed=42):
    """합성 폴백 — 실데이터와 동일한 (train, test, test_rul) 3종 세트 생성.

    train: 전 유닛이 고장 시점까지 기록된 전체 궤적
    test : 고장 전 임의 시점에서 잘린 궤적
    test_rul: test 각 유닛의 '잘린 시점 기준 남은 사이클' (RUL_FDxxx.txt 대응)
    """
    full = mfg_datagen.gen_cmapss_like(n_units=100, subset=subset, seed=seed)
    rng = np.random.default_rng(seed)
    units = full["unit"].unique()
    train_units = units[:70]
    test_units = units[70:]
    train = full[full["unit"].isin(train_units)].reset_index(drop=True)

    test_parts, ruls = [], []
    for i, u in enumerate(test_units):
        g = full[full["unit"] == u]
        life = g["cycle"].max()
        cut = int(rng.integers(int(life * 0.4), int(life * 0.9)))
        g = g[g["cycle"] <= cut].copy()
        g["unit"] = i + 1                       # 실데이터처럼 1부터 재번호
        test_parts.append(g)
        ruls.append(life - cut)
    test = pd.concat(test_parts).reset_index(drop=True)
    test_rul = pd.Series(ruls, name="RUL")
    # train 유닛도 1부터 재번호
    remap = {u: i + 1 for i, u in enumerate(train_units)}
    train["unit"] = train["unit"].map(remap)
    return train, test, test_rul


def load_cmapss(subset="FD001", fallback=True):
    """NASA C-MAPSS 로드. 반환: (train_df, test_df, test_rul)

    train_df/test_df 컬럼: unit, cycle, op1..op3, s1..s21
    test_rul: test 유닛별 실제 잔여수명(Series). RUL 라벨은 add_rul()로 붙인다.
    """
    raw_tr = _fetch(f"cmapss/train_{subset}.txt")
    raw_te = _fetch(f"cmapss/test_{subset}.txt")
    raw_ru = _fetch(f"cmapss/RUL_{subset}.txt")
    if raw_tr and raw_te and raw_ru:
        train = _parse_cmapss_txt(raw_tr)
        test = _parse_cmapss_txt(raw_te)
        test_rul = pd.read_csv(io.BytesIO(raw_ru), header=None).iloc[:, 0].rename("RUL")
        print(f"✅ C-MAPSS {subset} 실데이터 로드 완료 "
              f"(train {train.shape}, test {test.shape})")
        return train, test, test_rul
    if not fallback:
        raise RuntimeError(f"C-MAPSS {subset} 다운로드 실패 (fallback=False)")
    print(FALLBACK_MSG)
    train, test, test_rul = _synth_cmapss(subset)
    print(f"   합성 {subset}: train {train.shape}, test {test.shape}")
    return train, test, test_rul


def add_rul(df, cap=125):
    """유닛별 RUL 라벨 부여: max(cycle) - cycle, 상한 cap으로 클리핑.

    초기 구간은 열화가 없어 '정확한 남은 수명'을 알 수 없으므로
    관례적으로 일정 값(cap)으로 눌러 학습을 안정화한다(piecewise linear RUL).
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


# ================================================================ 다변량 이상탐지
# SKAB(수순환 펌프 테스트베드)의 8개 센서명 — 실데이터/합성 폴백 공통 스키마
ANOMALY_SENSORS = ["accel1_rms", "accel2_rms", "current", "pressure",
                   "temperature", "thermocouple", "voltage", "flow_rate"]


def load_anomaly(fallback=True):
    """다변량 센서 이상탐지 데이터 로드. 반환: (train_df, test_df, test_labels)

    train_df: 정상 가동 구간 (One-class 학습용), 컬럼 ANOMALY_SENSORS 8개
    test_df : 평가 구간 (이상 포함), 동일 컬럼
    test_labels: 시점별 0/1 라벨 (Series)
    실데이터는 SKAB 펌프 벤치마크를 upload/prep_skab.py 로 가공해 올려둔다(운영가이드 참조).
    """
    raw_tr = _fetch("anomaly/train.csv")
    raw_te = _fetch("anomaly/test.csv")
    raw_la = _fetch("anomaly/test_labels.csv")
    if raw_tr and raw_te and raw_la:
        train = pd.read_csv(io.BytesIO(raw_tr))
        test = pd.read_csv(io.BytesIO(raw_te))
        labels = pd.read_csv(io.BytesIO(raw_la)).iloc[:, 0].rename("label")
        print(f"✅ 이상탐지 실데이터 로드 완료 (train {train.shape}, test {test.shape}, "
              f"이상 비율 {100 * labels.mean():.2f}%)")
        return train, test, labels
    if not fallback:
        raise RuntimeError("이상탐지 데이터 다운로드 실패 (fallback=False)")
    print(FALLBACK_MSG)
    X, y, _ = mfg_datagen.gen_multivar_anomaly(n=8000, n_sensors=8, seed=42)
    X.columns = ANOMALY_SENSORS
    split = int(len(X) * 0.55)
    train = X.iloc[:split][y.iloc[:split] == 0].reset_index(drop=True)   # 정상만
    test = X.iloc[split:].reset_index(drop=True)
    labels = y.iloc[split:].reset_index(drop=True)
    print(f"   합성: train {train.shape}, test {test.shape}, "
          f"이상 비율 {100 * labels.mean():.2f}%")
    return train, test, labels


# ================================================================ 펌프 설비 실데이터
PUMP_SENSORS = ["sensor_01", "sensor_03", "sensor_04", "sensor_05", "sensor_10",
                "sensor_12", "sensor_28", "sensor_36", "sensor_48",
                "sensor_15", "sensor_50", "sensor_51"]


def _synth_pump(seed=42):
    """펌프 실데이터 스키마 모사 폴백 — 결측·죽은 센서·고장 이벤트까지 재현."""
    rng = np.random.default_rng(seed)
    n = 44064                                            # 5분 × 153일
    idx = pd.date_range("2018-04-01", periods=n, freq="5min")
    t = np.arange(n)

    # 공통 잠재 신호(펌프 부하) + 센서별 반응
    load = 1.0 + 0.25 * np.sin(2 * np.pi * t / 288) + 0.1 * np.sin(2 * np.pi * t / 2016)
    df = pd.DataFrame(index=idx)
    base = rng.uniform(20, 120, len(PUMP_SENSORS))
    for j, c in enumerate(PUMP_SENSORS):
        df[c] = base[j] * (load ** rng.uniform(0.5, 1.5)) + rng.normal(0, base[j] * 0.02, n)

    # 고장 7건 — 직전 완만한 이탈 + 정지 + 회복 구간
    status = np.array(["NORMAL"] * n, dtype=object)
    fails = np.sort(rng.choice(np.arange(int(n * 0.05), int(n * 0.95)), 7, replace=False))
    for f in fails:
        pre = slice(max(0, f - 36), f)                   # 고장 3시간 전 미약한 전조
        for c in PUMP_SENSORS[:5]:
            df.loc[df.index[pre], c] *= rng.uniform(0.95, 0.98)
        rec = slice(f, min(n, f + int(rng.integers(200, 600))))
        df.loc[df.index[rec], PUMP_SENSORS] *= 0.05      # 정지 — 값이 바닥으로
        status[rec] = "RECOVERING"
        status[f] = "BROKEN"
    df["machine_status"] = status

    # 실데이터의 결측 구조 재현
    df[PUMP_SENSORS[9]] = np.nan                                        # sensor_15: 죽은 센서
    m50 = rng.random(n) < 0.35
    df.loc[m50, PUMP_SENSORS[10]] = np.nan                              # sensor_50: 35% 결측
    df = mfg_datagen._inject_missing_blocks(
        df, [PUMP_SENSORS[11]], rng, n_blocks=6, block_len=(200, 600))
    return df.reset_index(names="timestamp")


def load_pump(fallback=True):
    """수처리장 펌프 설비 데이터. 반환: DataFrame

    컬럼: timestamp, sensor_01 ... sensor_51 (12개), machine_status
    machine_status: NORMAL / RECOVERING / BROKEN
    원본은 1분 샘플링 220,320행이며, 수업용으로 5분 평균 44,064행으로 줄여 배포한다.
    결측은 일부러 채우지 않았다 — 죽은 센서와 통신 두절 블록이 실습 소재다.
    """
    raw = _fetch("pump/pump_5min.csv")
    if raw is not None:
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["timestamp"])
        n_brk = int((df["machine_status"] == "BROKEN").sum())
        print(f"✅ 펌프 실데이터 로드 완료 {df.shape} | 고장 {n_brk}건")
        return df
    if not fallback:
        raise RuntimeError("펌프 데이터 다운로드 실패 (fallback=False)")
    print(FALLBACK_MSG)
    df = _synth_pump()
    n_brk = int((df["machine_status"] == "BROKEN").sum())
    print(f"   합성 펌프: {df.shape} | 고장 {n_brk}건")
    return df


# ================================================================ 체크포인트
def load_checkpoint(name, map_location="cpu"):
    """사전학습 체크포인트 로드. 반환: state_dict 또는 None(→ 데모 모델 폴백).

    사용 예:
        sd = load_checkpoint("d1_03_anomaly_transformer")
        if sd is not None:
            model.load_state_dict(sd)
        else:
            pass  # 방금 학습한 데모 모델 그대로 사용
    """
    raw = _fetch(f"checkpoints/{name}.pt")
    if raw is None:
        print(CKPT_MSG)
        return None
    try:
        import torch
        sd = torch.load(io.BytesIO(raw), map_location=map_location, weights_only=True)
        print(f"✅ 체크포인트 로드 완료: {name}")
        return sd
    except Exception:
        print(CKPT_MSG)
        return None


if __name__ == "__main__":
    # 폴백 경로 자가 검증 (네트워크 차단 상태 가정)
    os.environ["MFG_FORCE_FALLBACK"] = "1"
    tr, te, rul = load_cmapss("FD001")
    assert list(tr.columns) == _CMAPSS_COLS and list(te.columns) == _CMAPSS_COLS
    assert len(rul) == te["unit"].nunique()
    tr_r = add_rul(tr)
    assert tr_r["RUL"].max() == 125 and (tr_r.groupby("unit")["RUL"].min() == 0).all()
    tr3, te3, _ = load_cmapss("FD003")
    a_tr, a_te, a_la = load_anomaly()
    assert list(a_tr.columns) == list(a_te.columns) == ANOMALY_SENSORS
    assert len(a_te) == len(a_la)
    p = load_pump()
    assert list(p.columns) == ["timestamp"] + PUMP_SENSORS + ["machine_status"]
    assert int((p["machine_status"] == "BROKEN").sum()) == 7
    assert p["sensor_15"].isna().all()                     # 죽은 센서 재현
    assert 0.2 < p["sensor_50"].isna().mean() < 0.5        # 결측 비율 재현
    assert load_checkpoint("no_such_ckpt") is None
    print("loaders OK (fallback path)")
