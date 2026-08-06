"""제조 합성 데이터 생성기 — Day 1 실습 공용.

물리적으로 그럴듯한 신호를 만든다. 모든 생성기 공통 요소:
  1. 트렌드(설비 열화·공구 마모)  2. 주기성(사이클·교대조·요일)
  3. 결측 블록(통신 두절, NaN 연속 구간)  4. 홀드값(센서 고착 — 동일값 반복)
  5. 이상 3종: Point(스파이크) / Contextual(정상 범위지만 맥락상 이상) / Collective(구간 패턴 이상)
  6. 라벨 극심 불균형(이상 비율 0.5~2%)  7. seed 인자, return_labels 옵션

C-MAPSS·SMAP 모사 생성기는 실데이터 로더(loaders.py)의 폴백으로 쓰이며
반환 스키마가 실데이터와 완전히 동일하다.
"""
from collections import Counter

import numpy as np
import pandas as pd

__all__ = [
    "gen_injection_molding", "gen_cnc_spindle", "gen_line_energy",
    "gen_cmapss_like", "gen_multivar_anomaly",
]


# ---------------------------------------------------------------- 공통 유틸
def _inject_missing_blocks(df, cols, rng, n_blocks=2, block_len=(20, 60)):
    """통신 두절을 흉내 낸 NaN 연속 블록을 심는다."""
    n = len(df)
    for _ in range(n_blocks):
        start = rng.integers(int(n * 0.1), int(n * 0.85))
        length = rng.integers(*block_len)
        df.iloc[start:start + length, [df.columns.get_loc(c) for c in cols]] = np.nan
    return df


def _inject_hold(df, col, rng, n_blocks=1, block_len=(30, 80)):
    """센서 고착(홀드값) — 마지막 정상값이 그대로 반복되는 구간."""
    n = len(df)
    for _ in range(n_blocks):
        start = rng.integers(int(n * 0.15), int(n * 0.8))
        length = int(rng.integers(*block_len))
        df.iloc[start:start + length, df.columns.get_loc(col)] = df.iloc[start][col]
    return df


def _dt_index(n, freq_sec, start="2025-03-03 08:00:00"):
    return pd.date_range(start, periods=n, freq=f"{freq_sec}s")


# ---------------------------------------------------------------- 1) 사출성형
def gen_injection_molding(n_cycles=120, cycle_sec=30, seed=42, return_labels=False):
    """사출성형기 센서 시계열 (1초 샘플링).

    - mold_temp   : 금형 온도(℃). 가동 초반 warm-up 램프 + 사이클 내 소폭 진동
    - inj_pressure: 사출압력(bar). 사이클마다 보압 피크
    - screw_rpm   : 스크류 회전수. 사출 구간에만 상승
    - 사이클타임 변동: 사이클 길이가 ±10% 요동

    이상 3종 주입(라벨 비율 약 1~2%):
      Point       — 압력 스파이크(순간 과충전)
      Contextual  — 냉각 구간인데 온도가 상승(냉각수 이상. 값 자체는 정상 범위)
      Collective  — 사이클 리듬 붕괴 구간(압력 파형이 낮고 불규칙)
    """
    rng = np.random.default_rng(seed)
    seg_lens, press, rpm, phase = [], [], [], []
    for _ in range(n_cycles):
        L = max(20, int(cycle_sec * (1 + rng.normal(0, 0.05))))  # 사이클타임 변동
        t = np.linspace(0, 1, L)
        p = 40 + 90 * np.exp(-((t - 0.25) ** 2) / 0.006)          # 사출 피크
        p += 25 * ((t > 0.3) & (t < 0.6))                          # 보압 구간
        r = 20 + 160 * ((t > 0.05) & (t < 0.35))                   # 사출 구간 회전
        seg_lens.append(L)
        press.append(p + rng.normal(0, 2.0, L))
        rpm.append(r + rng.normal(0, 4.0, L))
        phase.append(t)
    press = np.concatenate(press)
    rpm = np.concatenate(rpm)
    phase = np.concatenate(phase)
    n = len(press)

    warmup = 45 + 20 * (1 - np.exp(-np.arange(n) / (n * 0.15)))    # warm-up 램프 → 65℃ 안정
    cyc_osc = 3.0 * np.sin(2 * np.pi * phase)                       # 사이클 내 온도 진동
    temp = warmup + cyc_osc + rng.normal(0, 0.4, n)

    labels = np.zeros(n, dtype=int)

    # Point — 압력 스파이크 4곳
    for i in rng.choice(np.arange(int(n * 0.2), int(n * 0.95)), 4, replace=False):
        press[i] += rng.uniform(60, 90)
        labels[i] = 1
    # Contextual — 냉각 구간(phase>0.6)에서 온도 상승. 절대값은 정상 범위(70℃ 미만)
    ctx = np.where(phase > 0.65)[0]
    ctx_start = ctx[rng.integers(int(len(ctx) * 0.5), int(len(ctx) * 0.7))]
    temp[ctx_start:ctx_start + 12] += 4.5
    labels[ctx_start:ctx_start + 12] = 1
    # Collective — 리듬 붕괴 구간
    col_start = rng.integers(int(n * 0.75), int(n * 0.85))
    L = 25
    press[col_start:col_start + L] = 55 + rng.normal(0, 12, L)
    rpm[col_start:col_start + L] = 60 + rng.normal(0, 20, L)
    labels[col_start:col_start + L] = 1

    df = pd.DataFrame(
        {"mold_temp": temp, "inj_pressure": press, "screw_rpm": rpm},
        index=_dt_index(n, 1),
    )
    df = _inject_missing_blocks(df, ["inj_pressure"], rng, n_blocks=1, block_len=(25, 45))
    df = _inject_hold(df, "mold_temp", rng, n_blocks=1, block_len=(40, 70))
    if return_labels:
        return df, pd.Series(labels, index=df.index, name="label")
    return df


# ---------------------------------------------------------------- 2) CNC 스핀들
def gen_cnc_spindle(n_hours=48, freq_sec=10, seed=42, return_labels=False):
    """CNC 스핀들 진동 RMS·부하 시계열.

    - vib_rms     : 진동 RMS(mm/s). 회전 주파수 성분 + 공구 마모에 따른 진폭 증가 추세
    - spindle_load: 스핀들 부하(%). 가공/비가공 주기
    - tool_age    : 공구 사용 시간(분). 교체 시 0으로 리셋
    """
    rng = np.random.default_rng(seed)
    n = int(n_hours * 3600 / freq_sec)
    t = np.arange(n)

    tool_life = int(8 * 3600 / freq_sec)                 # 공구 수명 약 8시간
    tool_age = (t % tool_life).astype(float)
    wear = 0.8 * (tool_age / tool_life) ** 1.5           # 마모 → 진폭 증가, 교체 시 리셋

    rot = 0.35 * np.sin(2 * np.pi * t / 6.0)             # 회전 주파수 성분
    machining = (np.sin(2 * np.pi * t / (1800 / freq_sec)) > -0.3).astype(float)  # 가공 주기
    vib = 0.9 + wear + np.abs(rot) * machining + rng.normal(0, 0.06, n)
    load = 25 + 45 * machining + 8 * wear + rng.normal(0, 2.5, n)

    labels = np.zeros(n, dtype=int)
    # Point — 충돌성 진동 스파이크
    for i in rng.choice(np.arange(int(n * 0.1), int(n * 0.95)), 5, replace=False):
        vib[i] += rng.uniform(2.5, 4.0)
        labels[i] = 1
    # Contextual — 비가공 구간인데 부하가 정상 가공 수준
    idle = np.where(machining < 0.5)[0]
    s = idle[rng.integers(len(idle) // 2, len(idle) - 40)]
    load[s:s + 30] = 60 + rng.normal(0, 2, 30)
    labels[s:s + 30] = 1
    # Collective — 베어링 이상: 진동 리듬 자체가 흐트러짐
    cs = rng.integers(int(n * 0.7), int(n * 0.82))
    L = 60
    vib[cs:cs + L] = 1.6 + 0.7 * np.sin(2 * np.pi * np.arange(L) / 2.3) + rng.normal(0, 0.25, L)
    labels[cs:cs + L] = 1

    df = pd.DataFrame(
        {"vib_rms": vib, "spindle_load": load, "tool_age_min": tool_age * freq_sec / 60},
        index=_dt_index(n, freq_sec),
    )
    df = _inject_missing_blocks(df, ["vib_rms", "spindle_load"], rng, n_blocks=2, block_len=(30, 60))
    df = _inject_hold(df, "spindle_load", rng, n_blocks=1, block_len=(50, 90))
    if return_labels:
        return df, pd.Series(labels, index=df.index, name="label")
    return df


# ---------------------------------------------------------------- 3) 라인 전력
def gen_line_energy(n_days=28, freq_min=10, seed=42, return_labels=False):
    """생산라인 전력 사용량(kW). 주/야간 교대조 주기 + 요일 효과 + 가동/정지."""
    rng = np.random.default_rng(seed)
    n = int(n_days * 24 * 60 / freq_min)
    idx = _dt_index(n, freq_min * 60, start="2025-03-03 00:00:00")
    hour = idx.hour + idx.minute / 60
    dow = idx.dayofweek

    base = 120 + 60 * ((hour >= 8) & (hour < 20))         # 주간조 부하
    base = base + 25 * ((hour >= 20) | (hour < 4))         # 야간조 감산 운전
    weekend = np.where(dow >= 5, -70, 0)                    # 주말 감산
    drift = np.linspace(0, 12, n)                           # 설비 노후화로 완만한 상승 트렌드
    power = base + weekend + drift + rng.normal(0, 6, n)
    power = np.clip(power, 15, None)

    # 계획 정지(PM) 구간 — 정상 이벤트
    pm = rng.integers(int(n * 0.4), int(n * 0.5))
    power[pm:pm + int(6 * 60 / freq_min)] = rng.normal(18, 1.5, int(6 * 60 / freq_min))

    labels = np.zeros(n, dtype=int)
    # Point — 순간 과부하
    for i in rng.choice(np.arange(int(n * 0.1), int(n * 0.95)), 4, replace=False):
        power[i] += rng.uniform(90, 130)
        labels[i] = 1
    # Contextual — 일요일 심야에 주간 수준 전력(무단 가동/누전. 값은 정상 범위)
    sun_night = np.where((dow == 6) & (hour < 5))[0]
    if len(sun_night) > 20:
        s = sun_night[len(sun_night) // 2]
        power[s:s + 12] = 175 + rng.normal(0, 4, 12)
        labels[s:s + 12] = 1
    # Collective — 리듬 어긋남: 교대 주기가 사라진 구간 (6시간)
    cs = rng.integers(int(n * 0.75), int(n * 0.85))
    L = int(6 * 60 / freq_min)
    power[cs:cs + L] = 140 + rng.normal(0, 5, L)
    labels[cs:cs + L] = 1

    df = pd.DataFrame({"power_kw": power}, index=idx)
    df = _inject_missing_blocks(df, ["power_kw"], rng, n_blocks=2, block_len=(6, 18))
    df = _inject_hold(df, "power_kw", rng, n_blocks=1, block_len=(12, 30))
    if return_labels:
        return df, pd.Series(labels, index=df.index, name="label")
    return df


# ---------------------------------------------------------------- 4) C-MAPSS 모사
_CMAPSS_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# 실데이터에서 확인되는 센서별 대략적 운전점(FD001 평균 근사)
_S_BASE = np.array([518.7, 642.0, 1585.0, 1400.0, 14.62, 21.6, 553.0, 2388.0,
                    9050.0, 1.3, 47.3, 521.0, 2388.0, 8130.0, 8.44, 0.03,
                    392.0, 2388.0, 100.0, 38.8, 23.3])
# 열화에 따라 움직이는 센서(실데이터에서 추세가 뚜렷한 채널)와 방향
_DRIFT_DIR = np.array([0, 1, 1, 1, 0, 0, -1, 0.2, 0.5, 0, 1, -1, 0.2, -0.5,
                       1, 0, 1, 0.2, 0, -1, -1], dtype=float)

def gen_cmapss_like(n_units=80, subset="FD001", seed=42, add_missing=False, add_hold=False):
    """NASA C-MAPSS 스키마 모사 폴백.

    반환: DataFrame[unit, cycle, op1, op2, op3, s1..s21] — 실데이터와 컬럼 동일.
    RUL은 실데이터와 마찬가지로 '고장까지 남은 사이클'로 후처리 계산한다(로더 참고).
    subset 파라미터가 운전조건을 바꿔 도메인 분기(FD001 vs FD003)를 만든다.
      - FD001: 단일 운전조건, 단일 고장모드(HPC 열화)
      - FD003: 운전점 오프셋 + 고장모드 2종 혼합 → 분포가 실제로 틀어진다
    add_missing/add_hold 는 기본 False (실데이터에 결측이 없어 스키마 일치 유지).
    """
    rng = np.random.default_rng(seed + (0 if subset == "FD001" else 7))
    rows = []
    # FD003 도메인 시프트: 운전점 오프셋 + 열화 경로 변화
    op_shift = 0.0 if subset == "FD001" else 0.4
    for u in range(1, n_units + 1):
        life = int(rng.integers(130, 300))
        t = np.arange(1, life + 1)
        frac = t / life
        # 고장모드: FD001은 HPC 단일, FD003은 HPC/Fan 혼합
        if subset == "FD003" and rng.random() < 0.5:
            degr = 1.6 * frac ** 1.7      # Fan 열화 — 더 이른 시점부터 진행
        else:
            degr = 1.2 * frac ** 2.2      # HPC 열화 — 말기 가속
        unit_bias = rng.normal(0, 0.05, 21)
        s = (_S_BASE[None, :]
             + _DRIFT_DIR[None, :] * degr[:, None] * (0.004 * _S_BASE)[None, :]
             + (_S_BASE * 0.0012)[None, :] * rng.standard_normal((life, 21))
             + (_S_BASE * unit_bias * 0.001)[None, :])
        op1 = rng.normal(0.0 + op_shift, 0.003, life)
        op2 = rng.normal(0.0 + op_shift * 0.5, 0.0003, life)
        op3 = np.full(life, 100.0)
        block = np.column_stack([np.full(life, u), t, op1, op2, op3, s])
        rows.append(block)
    df = pd.DataFrame(np.vstack(rows), columns=_CMAPSS_COLS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    if add_missing:
        rng2 = np.random.default_rng(seed)
        df = _inject_missing_blocks(df, ["s2", "s7"], rng2, n_blocks=2, block_len=(30, 60))
    if add_hold:
        rng2 = np.random.default_rng(seed + 1)
        df = _inject_hold(df, "s11", rng2, n_blocks=1, block_len=(40, 80))
    return df


# ---------------------------------------------------------------- 5) 다변량 이상탐지
# 압출기(Extruder) 8채널 — 토의 세션의 상황카드와 같은 설비를 쓴다
EXTRUDER_SENSORS = ["barrel_temp1", "barrel_temp2", "die_pressure", "screw_torque",
                    "screw_rpm", "melt_temp", "feed_rate", "cooling_flow"]


def gen_multivar_anomaly(n=8000, n_sensors=8, seed=42, anomaly_ratio=0.015,
                         sensor_names=None):
    """다변량 상관 시계열 + 시점별 이상 라벨 + 원인 센서 라벨.

    반환: (df, labels: 0/1 Series, root_cause: dict)
      - 센서 간 상관: 공통 잠재 신호 2개를 섞어 생성 (한 센서가 흔들리면 이웃도 흔들린다)
      - 이상 구간마다 **원인 센서가 하나** → root_cause["point_root"]로 시점별 원인을 준다
      - 라벨 비율 anomaly_ratio(기본 1.5%) — 극심 불균형
      - 이상 3종: Point(스파이크) / Contextual(정상 범위지만 맥락상 이상) / Collective(구간 패턴 이상)

    root_cause 구조:
      {"segments": [(start, length, kind, sensor_name), ...],
       "point_root": np.array(길이 n, dtype=object) — 정상 시점은 None}
    """
    rng = np.random.default_rng(seed)
    if sensor_names is None:
        sensor_names = (EXTRUDER_SENSORS[:n_sensors] if n_sensors <= len(EXTRUDER_SENSORS)
                        else [f"s{i + 1:02d}" for i in range(n_sensors)])
    cols = list(sensor_names)
    t = np.arange(n)
    z1 = np.sin(2 * np.pi * t / 240) + 0.3 * np.sin(2 * np.pi * t / 37)   # 공정 사이클
    z2 = np.sin(2 * np.pi * t / 720 + 1.2)                                # 교대조 리듬
    trend = np.linspace(0, 0.25, n)                                        # 완만한 열화
    W = rng.uniform(0.3, 1.0, (2, n_sensors)) * rng.choice([-1, 1], (2, n_sensors))
    X = np.outer(z1, W[0]) + np.outer(z2, W[1]) + trend[:, None] \
        + 0.12 * rng.standard_normal((n, n_sensors))

    labels = np.zeros(n, dtype=int)
    point_root = np.full(n, None, dtype=object)
    segments = []
    # 유형별 개수를 먼저 배분한다 — 세 유형이 고루 등장해야 교육이 성립한다
    budget = int(n * anomaly_ratio)
    n_point = max(4, budget // 12)                    # Point는 1시점이라 개수로 확보
    n_ctx = max(2, budget // 60)
    n_col = max(2, budget // 70)
    # 유형마다 '각각' 시계열 전 구간에 균등 분포시킨 뒤 합친다.
    # 무작위로 뿌리거나 단순 교대 배치를 하면 개수가 적은 유형이 한쪽에 몰려,
    # 학습/평가를 시간으로 나눌 때 특정 유형이 평가 구간에서 사라지는 사고가 난다.
    lo, hi = int(n * 0.08), int(n * 0.94)
    span = hi - lo
    plan = []
    for kind, cnt in [("point", n_point), ("contextual", n_ctx), ("collective", n_col)]:
        for i in range(cnt):
            plan.append((lo + int(span * (i + 0.5) / cnt), kind))
    plan.sort()
    slot_w = span / max(len(plan), 1)

    for slot, kind in plan:
        c = int(rng.integers(0, n_sensors))                     # 원인 센서
        placed = False
        for _ in range(40):                                     # 슬롯 안에서 겹치지 않는 자리 찾기
            jitter = int(rng.integers(-slot_w * 0.35, slot_w * 0.35 + 1))
            start = int(np.clip(slot + jitter, lo, hi))
            if not labels[max(0, start - 60): start + 60].any():
                placed = True
                break
        if not placed:
            continue
        if kind == "point":
            X[start, c] += rng.choice([-1, 1]) * rng.uniform(3.0, 4.5)
            L = 1
        elif kind == "contextual":
            L = int(rng.integers(15, 30))
            # 값 자체는 정상 범위지만 이웃 센서와의 관계가 깨진다 (위상 반전)
            X[start:start + L, c] = -X[start:start + L, c] + 2 * X[:, c].mean()
        else:
            L = int(rng.integers(25, 45))
            # 리듬 자체가 어긋난 구간 — 원래 주기보다 훨씬 빠른 진동
            X[start:start + L, c] = X[start, c] + 0.05 * rng.standard_normal(L) \
                + 1.2 * np.sin(2 * np.pi * np.arange(L) / 5.0)
        labels[start:start + L] = 1
        point_root[start:start + L] = cols[c]
        segments.append((start, L, kind, cols[c]))
    segments.sort()
    df = pd.DataFrame(X, columns=cols)
    return (df, pd.Series(labels, name="label"),
            {"segments": segments, "point_root": point_root})


if __name__ == "__main__":
    # 간단 자가 검증
    for name, fn in [("injection", gen_injection_molding), ("cnc", gen_cnc_spindle),
                     ("energy", gen_line_energy)]:
        df, y = fn(seed=42, return_labels=True)
        ratio = 100 * y.mean()
        nan_cnt = int(df.isna().sum().sum())
        print(f"{name:10s} shape={df.shape} 이상비율={ratio:.2f}% NaN={nan_cnt}")
        assert 0.3 <= ratio <= 2.5, name
        assert nan_cnt > 0, name
    d1 = gen_cmapss_like(n_units=20, subset="FD001", seed=1)
    d3 = gen_cmapss_like(n_units=20, subset="FD003", seed=1)
    print("cmapss_like FD001", d1.shape, "| FD003", d3.shape, "| cols ok:",
          list(d1.columns) == _CMAPSS_COLS)
    X, y, rc = gen_multivar_anomaly(seed=0)
    kinds = Counter(k for _, _, k, _ in rc["segments"])
    print(f"multivar   shape={X.shape} 이상비율={100 * y.mean():.2f}% "
          f"구간={len(rc['segments'])} 유형={dict(kinds)}")
    assert set(kinds) == {"point", "contextual", "collective"}, kinds
    assert (rc["point_root"] != None).sum() == int(y.sum())          # noqa: E711
    print("OK")
