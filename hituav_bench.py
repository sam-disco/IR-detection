# -*- coding: utf-8 -*-
"""
================================================================================
 HIT-UAV : YOLO11s vs YOLOv5s  —  학습/추론 시간 + 성능 비교 (간소화판)
================================================================================
 원본 9개 스크립트(step1c~step5) 중 그래프·분석 단계를 걷어내고,
 "데이터셋 구축 → 학습 → 추론 → 성능/속도 비교" 만 남긴 파일입니다.

 제거한 것 : step2/step2b(데이터 분석), step3c(교란 통제), step5(발표용 그림),
             모든 matplotlib 그림 생성
 남긴 것   : step1c(build), train11, eval11, patch5, train5, eval5
 추가한 것 : 학습 시간(에폭별 포함) · 추론 시간(전처리/추론/NMS 분해) 정밀 측정

--------------------------------------------------------------------------------
 사용법 (PowerShell)
--------------------------------------------------------------------------------
   python hituav_bench.py build          # HIT-UAV 원본 -> YOLO 구조 + COCO GT
   python hituav_bench.py setup5         # yolov5-7.0 내려받기 + 호환 패치
   python hituav_bench.py train11        # YOLO11s 학습   (시간 측정)
   python hituav_bench.py eval11         # YOLO11s 평가   (시간 측정)
   python hituav_bench.py train5         # YOLOv5s 학습   (시간 측정)
   python hituav_bench.py eval5          # YOLOv5s 평가   (시간 측정)
   python hituav_bench.py compare        # 두 모델 최종 비교표

   python hituav_bench.py all            # build~compare 전부 순차 실행
   python hituav_bench.py train5 --smoke # 3에폭 스모크 테스트

 시간 기록은 전부 meta/timing_log.csv 에 누적됩니다.
================================================================================
"""

import argparse
import collections
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import types
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 경로 · 설정
# ==============================================================================

# ------------------------------------------------------------------------------
# 모든 경로가 이 폴더 하나 아래에 있습니다. 다른 폴더는 참조하지 않습니다.
#
#   IR data_new/
#   ├── raw/            ← HIT-UAV 원본을 여기 넣어주세요 (직접 준비할 유일한 것)
#   ├── yolov5-7.0/     ← 'setup5' 단계가 자동으로 내려받습니다
#   ├── dataset/        ← build 가 생성
#   ├── meta/           ← 시간 기록 · COCO GT · 로그
#   ├── outputs/        ← 평가 결과
#   └── runs/           ← 학습된 가중치
# ------------------------------------------------------------------------------

ROOT = Path(r"C:\Users\User\Desktop\IR data_new")

RAW   = ROOT / "raw"            # HIT-UAV 원본 (normal_xml, normal_json 포함)
V5DIR = ROOT / "yolov5-7.0"     # YOLOv5 v7.0 저장소 (setup5 가 자동 생성)
DSD   = ROOT / "dataset"        # YOLO 학습용 구조
META  = ROOT / "meta"           # CSV / COCO GT / 로그
OUTD  = ROOT / "outputs"        # 평가 결과
RUNS  = ROOT / "runs"           # 학습 산출물(가중치)

# ---- 모델 설정 ---------------------------------------------------------------
V11_WEIGHTS  = "yolo11s.pt"
V11_RUN_NAME = "yolo11s_base"
V11_EPOCHS   = 100
V11_BATCH    = 16

V5_WEIGHTS   = "yolov5s.pt"
V5_RUN_NAME  = "yolov5s_base"
V5_EPOCHS    = 100
V5_BATCH     = 16

IMGSZ   = 640
SEED    = 0
WORKERS = 8

# ---- 평가 설정 (두 모델 완전 동일) --------------------------------------------
CONF_THR = 0.001     # AP 평가 표준값
NMS_IOU  = 0.7
MAX_DET  = 300

# ---- 속도 측정 설정 -----------------------------------------------------------
WARMUP_N   = 20      # 워밍업 이미지 수 (측정에서 제외)
SPEED_BS   = 1       # 지연시간(latency) 측정은 batch=1 이 기준

# ---- 데이터 정의 --------------------------------------------------------------
KEEP   = ["Person", "Car", "Bicycle", "OtherVehicle"]
IGNORE = "DontCare"
NAMES  = KEEP
EXPECT = {"Person": 12312, "Car": 7311, "Bicycle": 4980,
          "OtherVehicle": 148, "DontCare": 148}       # 논문 Fig.6(a)
SPLIT_EXPECT = (2029, 290, 579)
FNAME_RE = re.compile(r"^(\d)_(\d{2,3})_(\d{1,3})_(\d)_(\d+)$")

# COCO 크기 구간 (tiny/small 세분화)
AREA = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [16 ** 2, 32 ** 2],
        "medium": [32 ** 2, 96 ** 2], "large": [96 ** 2, 1e10]}
ALBL = list(AREA.keys())


# ==============================================================================
# 1. 공통 유틸
# ==============================================================================

_LOG = []
_OUT = OUTD


def set_out(sub):
    global _OUT, _LOG
    _OUT = OUTD / sub if sub != "." else META
    _OUT.mkdir(parents=True, exist_ok=True)
    _LOG = []
    return _OUT


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _LOG.append(s)


def sec(t):
    P("")
    P("=" * 70)
    P(t)
    P("=" * 70)


def die(msg):
    P("")
    P("!" * 70)
    P("[중단] " + str(msg))
    P("!" * 70)
    sys.exit(1)


def write_log(name):
    try:
        (_OUT / name).write_text("\n".join(_LOG), encoding="utf-8")
        P("  [log]", name)
    except Exception:
        pass


def dump(df, n):
    df.to_csv(_OUT / n, index=False, encoding="utf-8-sig")
    P("  [csv]", n)


def hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ 시간 기록

def names_for(smoke=False):
    """
    smoke 여부에 따라 실행 이름과 파일 접미사를 한 곳에서 결정.
    smoke 결과가 본 실행 결과를 덮어쓰지 않도록 이름을 분리합니다.
    """
    if smoke:
        return dict(v11="yolo11s_smoke", v5="yolov5s_smoke", tag="_smoke")
    return dict(v11=V11_RUN_NAME, v5=V5_RUN_NAME, tag="")


def record_timing(stage, seconds, note="", extra=None):
    """
    모든 단계 소요시간을 meta/timing_log.csv 에 누적.
    extra: {"epochs":100, "sec_per_epoch":12.3, ...} 같은 부가 정보(문자열로 저장)
    """
    META.mkdir(parents=True, exist_ok=True)
    f = META / "timing_log.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "stage", "seconds", "hh:mm:ss",
                        "device", "note", "detail"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), stage,
                    round(seconds, 3), hms(seconds), gpu_name(), note,
                    json.dumps(extra, ensure_ascii=False) if extra else ""])
    P(f"\n[TIME] {stage} : {seconds:.1f}초 ({seconds/60:.2f}분, {hms(seconds)})"
      f"  -> meta/timing_log.csv")


def save_speed(model_key, d):
    """추론 속도 상세를 meta/speed_<model>.json 에 저장 (compare 단계에서 읽음)"""
    META.mkdir(parents=True, exist_ok=True)
    (META / f"speed_{model_key}.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def save_train_stat(model_key, d):
    META.mkdir(parents=True, exist_ok=True)
    (META / f"train_{model_key}.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


# ------------------------------------------------------------------ COCO 평가

def coco_ap(e, area="all", iou=None):
    """COCOeval 결과에서 AP 추출. iou=None 이면 AP@[.5:.95]"""
    p = e.eval["precision"]                 # [T, R, K, A, M]
    ai, mi = ALBL.index(area), 2            # maxDets = MAX_DET
    if iou is None:
        s = p[:, :, :, ai, mi]
    else:
        ti = int(np.argmin(np.abs(e.params.iouThrs - iou)))
        s = p[ti:ti + 1, :, :, ai, mi]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else float("nan")


def make_evaluator(gt, dt):
    """COCOeval 실행기 생성 (두 모델 동일 설정)"""
    from pycocotools.cocoeval import COCOeval

    def ev(img_ids=None, cat_ids=None):
        e = COCOeval(gt, dt, "bbox")
        e.params.areaRng = [AREA[k] for k in ALBL]
        e.params.areaRngLbl = ALBL
        e.params.maxDets = [1, 10, MAX_DET]
        if img_ids is not None:
            e.params.imgIds = sorted(img_ids)
        if cat_ids is not None:
            e.params.catIds = list(cat_ids)
        e.evaluate()
        e.accumulate()
        return e

    return ev


def evaluate_common(gt, dt, out_prefix):
    """
    두 모델에 공통으로 적용하는 평가:
      전체 지표 + 크기별 AP + 클래스별 AP + 주야별
    반환: (전체지표 dict, 클래스별 DataFrame)
    """
    ev = make_evaluator(gt, dt)

    sec("전체 성능")
    E = ev()
    ov = dict(mAP50_95=coco_ap(E), mAP50=coco_ap(E, iou=.5), mAP75=coco_ap(E, iou=.75),
              AP_tiny=coco_ap(E, "tiny"), AP_small=coco_ap(E, "small"),
              AP_medium=coco_ap(E, "medium"), AP_large=coco_ap(E, "large"))
    for k, v in ov.items():
        P(f"  {k:<12} {v:.4f}")
    dump(pd.DataFrame([ov]).round(4), f"{out_prefix}_overall.csv")

    sec("클래스별 성능")
    rows = []
    for i, nm in enumerate(NAMES):
        e = ev(cat_ids=[i + 1])
        m50, m75 = coco_ap(e, iou=.5), coco_ap(e, iou=.75)
        rows.append(dict(클래스=nm, GT수=len(gt.getAnnIds(catIds=[i + 1])),
                         mAP50=round(m50, 4), mAP75=round(m75, 4),
                         mAP50_95=round(coco_ap(e), 4),
                         하락률=round((1 - m75 / max(m50, 1e-9)) * 100, 1)))
    cdf = pd.DataFrame(rows)
    P(cdf.to_string(index=False))
    dump(cdf, f"{out_prefix}_by_class.csv")

    return ov, cdf


def load_gt_and_map():
    """COCO GT + stem->image_id 매핑 로드"""
    from pycocotools.coco import COCO
    gtf = META / "coco_gt_test.json"
    if not gtf.exists():
        die(f"{gtf} 없음. 먼저 'python hituav_bench.py build' 를 실행하십시오.")
    gt = COCO(str(gtf))
    mp = load_json(META / "coco_image_id_map.json", {})
    stem2id = {k: int(v) for k, v in mp.get("stem_to_image_id", {}).items()}
    return gt, stem2id


# ------------------------------------------------------------------ v5 어댑터

def apply_v5_adapters():
    """
    YOLOv5 v7.0 은 유지보수 모드라 최신 torch/numpy 와 충돌.
    학습·평가 양쪽에 동일하게 적용해야 하는 우회 처리.
    """
    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    if not hasattr(getattr(torch.cuda, "amp", None), "GradScaler"):
        m = types.ModuleType("torch.cuda.amp")
        m.GradScaler = lambda **kw: torch.amp.GradScaler("cuda", **kw)
        m.autocast = lambda enabled=True, **kw: torch.amp.autocast("cuda", enabled=enabled, **kw)
        torch.cuda.amp = m
        sys.modules["torch.cuda.amp"] = m
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    return torch


def require_gpu():
    import torch
    P("python:", sys.version.split()[0], "| torch:", torch.__version__,
      "| cuda:", torch.cuda.is_available(), "|", gpu_name())
    if not torch.cuda.is_available():
        P("[경고] GPU 를 찾지 못했습니다. CPU 로 진행하면 매우 느립니다.")
    return torch.cuda.is_available()


# ==============================================================================
# STAGE 1 : build — HIT-UAV 원본 -> YOLO 구조 + data.yaml + 메타 CSV + COCO GT
# ==============================================================================
# 좌표는 normal_xml/Annotations (VOC, xmin/ymin/xmax/ymax) 에서만 읽습니다.
# HIT-UAV 의 normal_json bbox 는 [xc,yc,w,h] 중심좌표라 COCO 형식과 다릅니다.
# 분할은 normal_json/{train,val,test}/ 폴더 멤버십(원저자 분할)을 따릅니다.
# ==============================================================================

def stage_build(force=False):
    t0 = time.perf_counter()
    set_out(".")
    for d in (DSD, META, OUTD, RUNS):
        d.mkdir(parents=True, exist_ok=True)

    # 이미 만들어져 있으면 건너뜀
    if not force and (DSD / "data.yaml").exists():
        n = {s: len(list((DSD / "images" / s).glob("*.jpg")))
             for s in ("train", "val", "test")}
        if (n["train"], n["val"], n["test"]) == SPLIT_EXPECT:
            P(f"[skip] dataset 이 이미 있습니다 {n}  (다시 만들려면 --force)")
            return

    # ---------------------------------------------------------- 1. 원천 탐색
    sec("1. SOURCE")
    if not RAW.exists():
        die(f"raw 폴더 없음: {RAW}\n\n"
            f"  HIT-UAV 원본을 아래 위치에 넣어주십시오:\n"
            f"    {RAW}\n\n"
            f"  안에 normal_xml / normal_json 폴더가 있으면 됩니다.\n"
            f"  (하위 깊이는 상관없습니다 — 자동으로 찾습니다)")
    cand = [p for p in RAW.rglob("*") if p.is_dir() and p.name == "normal_xml"]
    if not cand:
        die(f"normal_xml 폴더를 {RAW} 아래에서 찾을 수 없습니다.\n"
            f"  raw 폴더 안에 HIT-UAV 원본이 제대로 들어있는지 확인하십시오.")
    SRC = cand[0].parent
    ANNO = SRC / "normal_xml" / "Annotations"
    JSPLIT = SRC / "normal_json"
    P("raw root    :", RAW)
    P("source root :", SRC)
    P("annotations :", ANNO, "->", len(list(ANNO.glob("*.xml"))), "xml")

    # ---------------------------------------------------------- 2. 분할 확보
    sec("2. SPLITS")
    splits = {}
    for s in ("train", "val", "test"):
        fs = sorted((JSPLIT / s).glob("*.jpg"))
        if not fs:
            die(f"normal_json/{s} 에 이미지가 없습니다.")
        splits[s] = [f.stem for f in fs]
        P(f"  {s:<6} {len(fs)}")
    allst = [x for v in splits.values() for x in v]
    if len(allst) != len(set(allst)):
        die("분할 간 중복 이미지 존재")
    got = (len(splits["train"]), len(splits["val"]), len(splits["test"]))
    if got != SPLIT_EXPECT:
        P(f"[경고] 분할 수가 {SPLIT_EXPECT} 와 다릅니다: {got} (계속 진행)")
    stem2split = {st: s for s, v in splits.items() for st in v}

    # ---------------------------------------------------------- 3. XML 파싱
    sec("3. PARSE VOC XML + VALIDATE")
    records, namecnt, badbox, noobj = {}, collections.Counter(), 0, 0
    for st in allst:
        xf = ANNO / f"{st}.xml"
        if not xf.exists():
            die(f"XML 누락: {xf}")
        r = ET.parse(xf).getroot()
        sz = r.find("size")
        W, H = int(float(sz.find("width").text)), int(float(sz.find("height").text))
        objs = []
        for ob in r.findall("object"):
            nm = ob.find("name").text.strip()
            namecnt[nm] += 1
            bb = ob.find("bndbox")
            x1, y1 = float(bb.find("xmin").text), float(bb.find("ymin").text)
            x2, y2 = float(bb.find("xmax").text), float(bb.find("ymax").text)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
                badbox += 1
                continue
            objs.append((nm, x1, y1, x2, y2))
        if not any(o[0] in KEEP for o in objs):
            noobj += 1
        records[st] = dict(W=W, H=H, objs=objs)

    P("클래스별 인스턴스 수 (파싱 vs 논문 Fig.6a):")
    for k in ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]:
        g, e = namecnt.get(k, 0), EXPECT[k]
        P(f"    [{'OK ' if g == e else 'DIFF'}] {k:<13} parsed={g:<6} paper={e}")
    P("총 박스:", sum(namecnt.values()), "(논문 24899)")
    P("퇴화/범위이탈 폐기:", badbox, "| 학습대상 0개 이미지:", noobj, "(배경으로 사용)")

    # ---------------------------------------------------------- 4. YOLO 구조
    sec("4. BUILD YOLO STRUCTURE")
    cls2idx = {c: i for i, c in enumerate(KEEP)}
    for s in splits:
        for sub in ("images", "labels"):
            p = DSD / sub / s
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    linked = copied = 0
    box_written = collections.Counter()
    for st, sp in stem2split.items():
        src_img = JSPLIT / sp / f"{st}.jpg"
        dst_img = DSD / "images" / sp / f"{st}.jpg"
        try:
            os.link(src_img, dst_img)          # 같은 드라이브면 용량 0
            linked += 1
        except Exception:
            shutil.copy2(src_img, dst_img)
            copied += 1
        rec = records[st]
        W, H = rec["W"], rec["H"]
        lines = []
        for nm, x1, y1, x2, y2 in rec["objs"]:
            if nm not in cls2idx:
                continue
            xc, yc = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{cls2idx[nm]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            box_written[nm] += 1
        (DSD / "labels" / sp / f"{st}.txt").write_text("\n".join(lines), encoding="utf-8")
    P(f"이미지 배치: hardlink {linked} / copy {copied}")
    P("YOLO 라벨 박스:", dict(box_written), "합계:", sum(box_written.values()))
    P(f"제외한 {IGNORE}:", namecnt.get(IGNORE, 0))

    # ---------------------------------------------------------- 5. data.yaml
    sec("5. data.yaml")
    fp = lambda p: str(p).replace("\\", "/")
    yml = (f'# HIT-UAV -> YOLO ({IGNORE} 제외)\n'
           f'path: "{fp(DSD)}"\n'
           f'train: "{fp(DSD / "images" / "train")}"\n'
           f'val: "{fp(DSD / "images" / "val")}"\n'
           f'test: "{fp(DSD / "images" / "test")}"\n\n'
           f'nc: {len(KEEP)}\nnames:\n' +
           "".join(f"  {i}: {c}\n" for i, c in enumerate(KEEP)))
    (DSD / "data.yaml").write_text(yml, encoding="utf-8")

    # YOLOv5 는 path + 상대경로 관례
    y5 = ('# HIT-UAV for YOLOv5 v7.0 (DontCare 제외)\n'
          f'path: {fp(DSD)}\n'
          'train: images/train\nval: images/val\ntest: images/test\n\n'
          f'nc: {len(KEEP)}\nnames:\n' +
          "".join(f"  {i}: {c}\n" for i, c in enumerate(KEEP)))
    (DSD / "data_v5.yaml").write_text(y5, encoding="utf-8")
    P("data.yaml / data_v5.yaml 생성 완료")

    # ---------------------------------------------------------- 6. 메타 CSV
    sec("6. METADATA CSV")
    rows = []
    for st, sp in stem2split.items():
        m = FNAME_RE.match(st)
        if not m:
            die(f"파일명 파싱 실패: {st}")
        t, alt, ang, w, sn = m.groups()
        rec = records[st]
        cc = collections.Counter(o[0] for o in rec["objs"])
        rows.append(dict(stem=st, split=sp,
                         time=("night" if t == "1" else "day"),
                         altitude=int(alt), angle=int(ang), weather=int(w), serial=int(sn),
                         width=rec["W"], height=rec["H"],
                         n_obj=sum(cc[c] for c in KEEP),
                         **{f"n_{c}": cc.get(c, 0) for c in KEEP}))
    with open(META / "image_meta.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    P("image_meta.csv :", len(rows), "행")

    # ---------------------------------------------------------- 7. COCO GT
    sec("7. COCO GT (test split)")
    coco = {"info": {"description": "HIT-UAV test split, DontCare excluded"},
            "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c, "supercategory": c}
                           for i, c in enumerate(KEEP)]}
    img_id_map, aid = {}, 1
    for i, st in enumerate(sorted(splits["test"])):
        rec = records[st]
        img_id_map[st] = i
        coco["images"].append({"id": i, "file_name": f"{st}.jpg",
                               "width": rec["W"], "height": rec["H"]})
        for nm, x1, y1, x2, y2 in rec["objs"]:
            if nm not in cls2idx:
                continue
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append({"id": aid, "image_id": i,
                                        "category_id": cls2idx[nm] + 1,
                                        "bbox": [round(x1, 2), round(y1, 2),
                                                 round(bw, 2), round(bh, 2)],
                                        "area": round(bw * bh, 2),
                                        "iscrowd": 0, "segmentation": []})
            aid += 1
    (META / "coco_gt_test.json").write_text(json.dumps(coco), encoding="utf-8")
    (META / "coco_image_id_map.json").write_text(
        json.dumps({"stem_to_image_id": img_id_map, "names": KEEP}, indent=2),
        encoding="utf-8")
    P("images:", len(coco["images"]), " annotations:", len(coco["annotations"]))

    areas = [a["area"] for a in coco["annotations"]]
    sm = sum(1 for a in areas if a < 32 ** 2)
    lg = sum(1 for a in areas if a > 96 ** 2)
    P(f"크기 분포(test): small {sm} ({sm/len(areas)*100:.1f}%) | "
      f"medium {len(areas)-sm-lg} | large {lg} ({lg/len(areas)*100:.1f}%)")

    # ---------------------------------------------------------- 8. 검증
    sec("8. FINAL VERIFY")
    ok = True
    for s in splits:
        ni = len(list((DSD / "images" / s).glob("*.jpg")))
        nl = len(list((DSD / "labels" / s).glob("*.txt")))
        good = (ni == nl == len(splits[s]))
        ok &= good
        P(f"  [{'OK ' if good else 'BAD'}] {s:<6} images={ni} labels={nl}")
    P(""); P(">>> RESULT:", "PASS" if ok else "CHECK NEEDED")

    write_log("build_log.txt")
    record_timing("build", time.perf_counter() - t0, f"{len(allst)} images")


# ==============================================================================
# STAGE 2 : setup5 — yolov5-7.0 확보 + 호환성 패치
# ==============================================================================

def _fetch_yolov5():
    """yolov5 v7.0 을 ROOT 안에 확보. git -> zip 순서로 시도."""
    ROOT.mkdir(parents=True, exist_ok=True)

    # (1) git clone
    try:
        subprocess.run(["git", "clone", "--branch", "v7.0", "--depth", "1",
                        "https://github.com/ultralytics/yolov5.git", str(V5DIR)],
                       check=True)
        P("  [OK] git clone 완료")
        return True
    except Exception as e:
        P(f"  [실패] git clone: {type(e).__name__} — zip 내려받기로 전환")

    # (2) zip 내려받기 (git 이 없어도 동작)
    try:
        import io, urllib.request, zipfile
        url = "https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip"
        P(f"  내려받는 중: {url}")
        with urllib.request.urlopen(url, timeout=180) as r:
            data = r.read()
        P(f"  {len(data)/1e6:.1f} MB 수신, 압축 해제 중...")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(ROOT)
        extracted = ROOT / "yolov5-7.0"          # zip 내부 폴더명이 이미 yolov5-7.0
        if not extracted.exists():
            cands = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("yolov5-")]
            if cands:
                cands[0].rename(V5DIR)
        P("  [OK] zip 설치 완료")
        return True
    except Exception as e:
        P(f"  [실패] zip 내려받기: {e}")
    return False


def stage_setup5():
    t0 = time.perf_counter()
    set_out(".")

    sec("1. yolov5-7.0 확인")
    if not V5DIR.exists():
        P(f"yolov5-7.0 없음 -> {V5DIR} 로 내려받기 시도")
        if not _fetch_yolov5():
            die(f"자동 설치 실패. 수동으로 받으십시오:\n\n"
                f"  git clone --branch v7.0 --depth 1 "
                f"https://github.com/ultralytics/yolov5.git \"{V5DIR}\"\n\n"
                f"  또는 아래 zip 을 받아 {ROOT} 에 풀고 폴더명을 yolov5-7.0 으로:\n"
                f"  https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip")
    P("yolov5 경로:", V5DIR)
    for f in ["train.py", "val.py", "utils/plots.py", "models/common.py"]:
        P(("  [OK] " if (V5DIR / f).exists() else "  [없음] ") + f)

    sec("2. Pillow 10+ 패치 (font.getsize -> getbbox)")
    pl = V5DIR / "utils" / "plots.py"
    src = pl.read_text(encoding="utf-8")
    hits = re.findall(r"\.getsize\(", src)
    P("발견된 .getsize( 호출:", len(hits))
    if hits:
        bak = pl.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy2(pl, bak)
            P("  백업:", bak.name)
        pl.write_text(re.sub(r"\.getsize\(([^)]*)\)", r".getbbox(\1)[2:]", src),
                      encoding="utf-8")
        P("  -> 치환 완료")
    else:
        P("  패치 불필요")

    sec("3. 런타임 점검")
    try:
        import pkg_resources  # noqa
        P("  [OK] pkg_resources")
    except Exception:
        P("  [경고] pkg_resources 없음 -> pip install \"setuptools<81\"")
    try:
        import torch
        P("  torch", torch.__version__, "| cuda", torch.cuda.is_available())
        P("  torch.cuda.amp.GradScaler:",
          "존재" if hasattr(getattr(torch.cuda, "amp", None), "GradScaler")
          else "없음 (shim 자동 적용)")
    except Exception as e:
        P("  torch 확인 실패:", e)

    write_log("setup5_log.txt")
    record_timing("setup5", time.perf_counter() - t0)


# ==============================================================================
# STAGE 3 : train11 — YOLO11s 학습 (에폭별 시간 기록)
# ==============================================================================

def stage_train11(smoke=False):
    t0 = time.perf_counter()
    set_out(".")
    from ultralytics import YOLO

    sec("YOLO11s 학습")
    has_gpu = require_gpu()
    N = names_for(smoke)
    epochs = 3 if smoke else V11_EPOCHS
    name, tag = N["v11"], N["tag"]
    P(f"weights={V11_WEIGHTS} epochs={epochs} batch={V11_BATCH} imgsz={IMGSZ}")
    P(f"저장 위치: {RUNS / name}")

    model = YOLO(V11_WEIGHTS)

    # ---- 에폭별 시간 측정 콜백 ----
    epoch_times = []
    state = {"t": None}

    def _on_epoch_start(trainer):
        state["t"] = time.perf_counter()

    def _on_epoch_end(trainer):
        if state["t"] is not None:
            dt = time.perf_counter() - state["t"]
            epoch_times.append(round(dt, 3))
            print(f"    [epoch {len(epoch_times)}] {dt:.1f}초")

    try:
        model.add_callback("on_train_epoch_start", _on_epoch_start)
        model.add_callback("on_train_epoch_end", _on_epoch_end)
    except Exception:
        P("[알림] 에폭 콜백 등록 실패 — 총 시간만 측정합니다.")

    t_train0 = time.perf_counter()
    model.train(data=str(DSD / "data.yaml").replace("\\", "/"),
                epochs=epochs, imgsz=IMGSZ, batch=V11_BATCH,
                device=0 if has_gpu else "cpu", workers=WORKERS, seed=SEED,
                project=str(RUNS).replace("\\", "/"), name=name,
                exist_ok=True, patience=30, plots=False)
    t_train = time.perf_counter() - t_train0

    n_ep = len(epoch_times) or epochs
    per_ep = (sum(epoch_times) / len(epoch_times)) if epoch_times else t_train / max(n_ep, 1)

    sec("학습 시간 요약")
    P(f"  총 학습시간   : {t_train:.1f}초  ({t_train/60:.2f}분, {hms(t_train)})")
    P(f"  완료 에폭     : {n_ep}")
    P(f"  에폭당 평균   : {per_ep:.2f}초")
    if epoch_times:
        P(f"  에폭 최소/최대: {min(epoch_times):.2f}초 / {max(epoch_times):.2f}초")
        pd.DataFrame({"epoch": range(1, len(epoch_times) + 1),
                      "seconds": epoch_times}).to_csv(
            META / f"epoch_times_yolo11{tag}.csv", index=False, encoding="utf-8-sig")
        P(f"  [csv] meta/epoch_times_yolo11{tag}.csv")

    save_train_stat(f"yolo11{tag}", dict(model="YOLO11s", weights=V11_WEIGHTS,
                                   epochs_requested=epochs, epochs_done=n_ep,
                                   batch=V11_BATCH, imgsz=IMGSZ,
                                   total_sec=round(t_train, 2),
                                   sec_per_epoch=round(per_ep, 3),
                                   device=gpu_name(), smoke=smoke))
    P("\n결과:", RUNS / name)
    write_log("train11_log.txt")
    record_timing("train11", time.perf_counter() - t0,
                  f"{epochs}ep batch{V11_BATCH}",
                  dict(epochs_done=n_ep, sec_per_epoch=round(per_ep, 3)))


# ==============================================================================
# STAGE 4 : eval11 — YOLO11s 추론(속도 측정) + COCO 평가
# ==============================================================================

def stage_eval11(smoke=False):
    t0 = time.perf_counter()
    N = names_for(smoke)
    name, tag = N["v11"], N["tag"]
    out = set_out(f"eval_{name}")

    sec("0. SETUP")
    # 무거운 import 전에 가중치부터 확인 (에러 메시지를 명확히)
    W = RUNS / name / "weights" / "best.pt"
    if not W.exists():
        other = RUNS / names_for(not smoke)["v11"] / "weights" / "best.pt"
        hint = (f"\n  참고: {other} 는 존재합니다. "
                f"{'--smoke 를 빼고' if smoke else '--smoke 를 붙여'} 실행해 보십시오.") \
            if other.exists() else ""
        die(f"best.pt 없음: {W}\n"
            f"  train11 을 {'--smoke 로 ' if smoke else ''}먼저 실행하십시오.{hint}")
    P("weights :", W)

    from ultralytics import YOLO
    import torch

    gt, stem2id = load_gt_and_map()
    P("test 이미지:", len(stem2id), "| GT 박스:", len(gt.getAnnIds()))

    model = YOLO(str(W))
    has_gpu = torch.cuda.is_available()
    dev = 0 if has_gpu else "cpu"

    # 파라미터 수
    try:
        n_param = sum(p.numel() for p in model.model.parameters())
    except Exception:
        n_param = None

    imgs = sorted((DSD / "images" / "test").glob("*.jpg"))
    if not imgs:
        die("test 이미지가 없습니다.")

    # ---------------------------------------------------------- 1. 워밍업
    sec(f"1. WARMUP ({WARMUP_N}장 — 측정에서 제외)")
    for p in imgs[:WARMUP_N]:
        model.predict(str(p), conf=CONF_THR, iou=NMS_IOU, max_det=MAX_DET,
                      imgsz=IMGSZ, device=dev, verbose=False)
    if has_gpu:
        torch.cuda.synchronize()
    P("  완료")

    # ---------------------------------------------------------- 2. 추론
    sec(f"2. INFERENCE (batch={SPEED_BS}, conf={CONF_THR}, NMS IoU={NMS_IOU})")
    dets = []
    sp_pre = sp_inf = sp_post = 0.0
    per_img = []
    t_all0 = time.perf_counter()
    for i, pth in enumerate(imgs, 1):
        ti = time.perf_counter()
        r = model.predict(str(pth), conf=CONF_THR, iou=NMS_IOU, max_det=MAX_DET,
                          imgsz=IMGSZ, device=dev, verbose=False)[0]
        if has_gpu:
            torch.cuda.synchronize()
        per_img.append((time.perf_counter() - ti) * 1000)

        sp_pre += r.speed.get("preprocess", 0.0)
        sp_inf += r.speed.get("inference", 0.0)
        sp_post += r.speed.get("postprocess", 0.0)

        iid = stem2id.get(pth.stem)
        if iid is None:
            continue
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xy = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        cl = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xy, cf, cl):
            dets.append({"image_id": int(iid), "category_id": int(k) + 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(c)})
        if i % 100 == 0:
            print(f"\r  {i}/{len(imgs)}", end="")
    print()
    t_all = time.perf_counter() - t_all0
    n = len(imgs)
    arr = np.array(per_img)

    sec("추론 속도 요약 — YOLO11s")
    P(f"  이미지 수        : {n}장 | 검출 {len(dets)}개")
    P(f"  총 추론시간      : {t_all:.2f}초")
    P(f"  장당 평균(wall)  : {arr.mean():.2f} ms   (중앙 {np.median(arr):.2f} / "
      f"p90 {np.percentile(arr,90):.2f})")
    P(f"  FPS              : {1000/arr.mean():.1f}")
    P(f"  내부 계측 분해   : 전처리 {sp_pre/n:.2f} / 추론 {sp_inf/n:.2f} / "
      f"NMS {sp_post/n:.2f} ms  (합 {(sp_pre+sp_inf+sp_post)/n:.2f})")

    speed = dict(model="YOLO11s", images=n, params=n_param,
                 total_sec=round(t_all, 3),
                 ms_per_image=round(float(arr.mean()), 3),
                 ms_median=round(float(np.median(arr)), 3),
                 ms_p90=round(float(np.percentile(arr, 90)), 3),
                 fps=round(1000 / float(arr.mean()), 2),
                 ms_preprocess=round(sp_pre / n, 3),
                 ms_inference=round(sp_inf / n, 3),
                 ms_postprocess=round(sp_post / n, 3),
                 device=gpu_name())
    save_speed(f"yolo11{tag}", speed)
    dump(pd.DataFrame([speed]), "speed.csv")

    # ---------------------------------------------------------- 3. 평가
    (out / "detections_test.json").write_text(json.dumps(dets), encoding="utf-8")
    dt = gt.loadRes(str(out / "detections_test.json"))
    ov, cdf = evaluate_common(gt, dt, "v11")
    (META / f"perf_yolo11{tag}.json").write_text(
        json.dumps({k: (None if np.isnan(v) else round(v, 4)) for k, v in ov.items()},
                   indent=2), encoding="utf-8")

    P(""); P("산출물:", out)
    write_log("eval11_log.txt")
    record_timing("eval11", time.perf_counter() - t0, f"{n} images, {len(dets)} dets",
                  dict(ms_per_image=speed["ms_per_image"], fps=speed["fps"]))


# ==============================================================================
# STAGE 5 : train5 — YOLOv5s 학습
# ==============================================================================

def stage_train5(smoke=False):
    t0 = time.perf_counter()
    set_out(".")
    cwd = os.getcwd()
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = ""
    if not V5DIR.exists():
        die(f"yolov5-7.0 없음: {V5DIR}  (setup5 를 먼저 실행)")

    torch = apply_v5_adapters()
    sec(f"YOLOv5s 학습  (SMOKE={smoke})")
    require_gpu()
    P("[patch] torch.load(weights_only=False) / cuda.amp shim / np.trapz 별칭 적용")

    try:
        os.chdir(V5DIR)
        sys.path.insert(0, str(V5DIR))
        import train as v5train
        P("[ok] yolov5 train 모듈 로드")

        N = names_for(smoke)
        epochs = 3 if smoke else V5_EPOCHS
        name, tag = N["v5"], N["tag"]
        cfg = dict(data=str(DSD / "data_v5.yaml").replace("\\", "/"),
                   weights=V5_WEIGHTS, imgsz=IMGSZ, batch_size=V5_BATCH,
                   device="0" if torch.cuda.is_available() else "cpu",
                   workers=WORKERS, seed=SEED, cache="ram",
                   project=str(RUNS).replace("\\", "/"), exist_ok=True,
                   epochs=epochs, name=name,
                   patience=100 if smoke else 30,
                   noval=False, noplots=True)
        for k, v in cfg.items():
            P(f"  {k:<12}= {v}")
        P("-" * 66)

        t_train0 = time.perf_counter()
        v5train.run(**cfg)
        t_train = time.perf_counter() - t_train0
    finally:
        os.chdir(cwd)

    # results.csv 로 에폭별 시간 역산
    per_ep, n_ep = None, epochs
    rc = RUNS / name / "results.csv"
    if rc.exists():
        try:
            df = pd.read_csv(rc)
            n_ep = len(df)
            per_ep = t_train / max(n_ep, 1)
            pd.DataFrame({"epoch": range(1, n_ep + 1),
                          "avg_seconds": [round(per_ep, 3)] * n_ep}).to_csv(
                META / f"epoch_times_yolov5{tag}.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
    if per_ep is None:
        per_ep = t_train / max(epochs, 1)

    sec("학습 시간 요약")
    P(f"  총 학습시간 : {t_train:.1f}초  ({t_train/60:.2f}분, {hms(t_train)})")
    P(f"  완료 에폭   : {n_ep}")
    P(f"  에폭당 평균 : {per_ep:.2f}초")

    save_train_stat(f"yolov5{tag}", dict(model="YOLOv5s", weights=V5_WEIGHTS,
                                   epochs_requested=epochs, epochs_done=n_ep,
                                   batch=V5_BATCH, imgsz=IMGSZ,
                                   total_sec=round(t_train, 2),
                                   sec_per_epoch=round(per_ep, 3),
                                   device=gpu_name(), smoke=smoke))
    P("\n결과:", RUNS / name)
    if smoke:
        P(">>> 스모크 통과. --smoke 없이 다시 실행하십시오.")
    write_log("train5_log.txt")
    record_timing("train5", time.perf_counter() - t0,
                  f"{epochs}ep batch{V5_BATCH} smoke={smoke}",
                  dict(epochs_done=n_ep, sec_per_epoch=round(per_ep, 3)))


# ==============================================================================
# STAGE 6 : eval5 — YOLOv5s 추론(속도 측정) + COCO 평가 (YOLO11 과 동일 조건)
# ==============================================================================

def stage_eval5(smoke=False):
    t0 = time.perf_counter()
    N = names_for(smoke)
    name, tag = N["v5"], N["tag"]
    out = set_out(f"eval_{name}")
    cwd = os.getcwd()
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    torch = apply_v5_adapters()

    sec("0. SETUP")
    WGT = RUNS / name / "weights" / "best.pt"
    if not WGT.exists():
        other = RUNS / names_for(not smoke)["v5"] / "weights" / "best.pt"
        hint = (f"\n  참고: {other} 는 존재합니다. "
                f"{'--smoke 를 빼고' if smoke else '--smoke 를 붙여'} 실행해 보십시오.") \
            if other.exists() else ""
        die(f"best.pt 없음: {WGT}\n"
            f"  train5 를 {'--smoke 로 ' if smoke else ''}먼저 실행하십시오.{hint}")
    P("weights :", WGT)

    try:
        os.chdir(V5DIR)
        sys.path.insert(0, str(V5DIR))
        from models.common import DetectMultiBackend
        from utils.dataloaders import LoadImages
        from utils.general import non_max_suppression, scale_boxes, check_img_size
        from utils.torch_utils import select_device

        device = select_device("0" if torch.cuda.is_available() else "cpu")
        model = DetectMultiBackend(str(WGT), device=device, dnn=False,
                                   data=str(DSD / "data_v5.yaml"), fp16=False)
        stride, pt = model.stride, model.pt
        imgsz = check_img_size((IMGSZ, IMGSZ), s=stride)
        model.warmup(imgsz=(1, 3, *imgsz))
        P("model loaded. stride =", stride)

        try:
            n_param = sum(p.numel() for p in model.model.parameters())
        except Exception:
            n_param = None

        gt, stem2id = load_gt_and_map()
        P("test:", len(stem2id), "| GT:", len(gt.getAnnIds()))

        # ------------------------------------------------------ 1. 워밍업
        sec(f"1. WARMUP ({WARMUP_N}장)")
        ds = LoadImages(str(DSD / "images" / "test"), img_size=imgsz,
                        stride=stride, auto=pt)
        wi = 0
        for path, im, im0s, _, _ in ds:
            t = torch.from_numpy(im).to(device).float() / 255.0
            if t.ndim == 3:
                t = t[None]
            pred = model(t, augment=False, visualize=False)
            non_max_suppression(pred, CONF_THR, NMS_IOU, None, False, max_det=MAX_DET)
            wi += 1
            if wi >= WARMUP_N:
                break
        if device.type != "cpu":
            torch.cuda.synchronize()
        P("  완료")

        # ------------------------------------------------------ 2. 추론
        sec(f"2. INFERENCE (conf={CONF_THR}, NMS IoU={NMS_IOU} — YOLO11 과 동일)")
        ds = LoadImages(str(DSD / "images" / "test"), img_size=imgsz,
                        stride=stride, auto=pt)
        dets, n = [], 0
        t_pre = t_fwd = t_nms = 0.0
        per_img = []
        t_all0 = time.perf_counter()
        for path, im, im0s, _, _ in ds:
            st = Path(path).stem
            if st not in stem2id:
                continue
            ti = time.perf_counter()

            _a = time.perf_counter()
            t = torch.from_numpy(im).to(device).float() / 255.0
            if t.ndim == 3:
                t = t[None]
            if device.type != "cpu":
                torch.cuda.synchronize()
            _b = time.perf_counter()

            pred = model(t, augment=False, visualize=False)
            if device.type != "cpu":
                torch.cuda.synchronize()
            _c = time.perf_counter()

            pred = non_max_suppression(pred, CONF_THR, NMS_IOU, None, False,
                                       max_det=MAX_DET)
            if device.type != "cpu":
                torch.cuda.synchronize()
            _d = time.perf_counter()

            t_pre += _b - _a
            t_fwd += _c - _b
            t_nms += _d - _c
            per_img.append((time.perf_counter() - ti) * 1000)

            for d in pred:
                if not len(d):
                    continue
                d[:, :4] = scale_boxes(t.shape[2:], d[:, :4], im0s.shape).round()
                for *xy, cf, cl in d.cpu().numpy():
                    x1, y1, x2, y2 = xy
                    dets.append({"image_id": stem2id[st], "category_id": int(cl) + 1,
                                 "bbox": [float(x1), float(y1),
                                          float(x2 - x1), float(y2 - y1)],
                                 "score": float(cf)})
            n += 1
            if n % 100 == 0:
                print(f"\r  {n}/{len(stem2id)}", end="")
        print()
        t_all = time.perf_counter() - t_all0
    finally:
        os.chdir(cwd)

    arr = np.array(per_img)
    sec("추론 속도 요약 — YOLOv5s")
    P(f"  이미지 수        : {n}장 | 검출 {len(dets)}개")
    P(f"  총 추론시간      : {t_all:.2f}초")
    P(f"  장당 평균(wall)  : {arr.mean():.2f} ms   (중앙 {np.median(arr):.2f} / "
      f"p90 {np.percentile(arr,90):.2f})")
    P(f"  FPS              : {1000/arr.mean():.1f}")
    P(f"  내부 계측 분해   : 전처리 {t_pre/n*1000:.2f} / 추론 {t_fwd/n*1000:.2f} / "
      f"NMS {t_nms/n*1000:.2f} ms")

    speed = dict(model="YOLOv5s", images=n, params=n_param,
                 total_sec=round(t_all, 3),
                 ms_per_image=round(float(arr.mean()), 3),
                 ms_median=round(float(np.median(arr)), 3),
                 ms_p90=round(float(np.percentile(arr, 90)), 3),
                 fps=round(1000 / float(arr.mean()), 2),
                 ms_preprocess=round(t_pre / n * 1000, 3),
                 ms_inference=round(t_fwd / n * 1000, 3),
                 ms_postprocess=round(t_nms / n * 1000, 3),
                 device=gpu_name())
    save_speed(f"yolov5{tag}", speed)
    dump(pd.DataFrame([speed]), "speed.csv")

    (out / "detections_test.json").write_text(json.dumps(dets), encoding="utf-8")
    dt = gt.loadRes(str(out / "detections_test.json"))
    ov, cdf = evaluate_common(gt, dt, "v5")
    (META / f"perf_yolov5{tag}.json").write_text(
        json.dumps({k: (None if np.isnan(v) else round(v, 4)) for k, v in ov.items()},
                   indent=2), encoding="utf-8")

    P(""); P("산출물:", out)
    write_log("eval5_log.txt")
    record_timing("eval5", time.perf_counter() - t0, f"{n} images, {len(dets)} dets",
                  dict(ms_per_image=speed["ms_per_image"], fps=speed["fps"]))


# ==============================================================================
# STAGE 7 : compare — 두 모델 최종 비교표 (시간 + 성능)
# ==============================================================================

def stage_compare(smoke=False):
    t0 = time.perf_counter()
    N = names_for(smoke)
    tag = N["tag"]
    out = set_out("compare" + tag)
    if smoke:
        P("[SMOKE] 3에폭 결과 비교입니다. 성능 수치는 의미 없고, "
          "파이프라인이 끝까지 도는지 확인하는 용도입니다.")

    tr11 = load_json(META / f"train_yolo11{tag}.json", {})
    tr5  = load_json(META / f"train_yolov5{tag}.json", {})
    sp11 = load_json(META / f"speed_yolo11{tag}.json", {})
    sp5  = load_json(META / f"speed_yolov5{tag}.json", {})
    pf11 = load_json(META / f"perf_yolo11{tag}.json", {})
    pf5  = load_json(META / f"perf_yolov5{tag}.json", {})

    missing = [n for n, d in [("train11", tr11), ("train5", tr5),
                              ("eval11", sp11), ("eval5", sp5)] if not d]
    if missing:
        P(f"[알림] 아직 실행되지 않은 단계: {', '.join(missing)} — 있는 것만 비교합니다.")

    # ---------------------------------------------------------- 시간 비교
    sec("1. 학습 / 추론 시간 비교")
    rows = []
    for key, tr, sp in [("YOLO11s", tr11, sp11), ("YOLOv5s", tr5, sp5)]:
        rows.append({
            "모델": key,
            "파라미터(M)": round(sp.get("params", 0) / 1e6, 2) if sp.get("params") else None,
            "에폭": tr.get("epochs_done"),
            "학습_총초": tr.get("total_sec"),
            "학습_분": round(tr["total_sec"] / 60, 2) if tr.get("total_sec") else None,
            "학습_hh:mm:ss": hms(tr["total_sec"]) if tr.get("total_sec") else None,
            "에폭당_초": tr.get("sec_per_epoch"),
            "추론_ms/장": sp.get("ms_per_image"),
            "추론_FPS": sp.get("fps"),
            "전처리_ms": sp.get("ms_preprocess"),
            "순추론_ms": sp.get("ms_inference"),
            "NMS_ms": sp.get("ms_postprocess"),
            "테스트_총초": sp.get("total_sec"),
        })
    tdf = pd.DataFrame(rows)
    P(tdf.to_string(index=False))
    dump(tdf, "compare_time.csv")

    if tr11.get("total_sec") and tr5.get("total_sec"):
        r = tr11["total_sec"] / tr5["total_sec"]
        faster = "YOLO11s" if r < 1 else "YOLOv5s"
        P(f"\n  학습 : {faster} 가 {max(r, 1/r):.2f}배 빠름 "
          f"(11s {tr11['total_sec']/60:.1f}분 vs v5s {tr5['total_sec']/60:.1f}분)")
    if sp11.get("ms_per_image") and sp5.get("ms_per_image"):
        r = sp11["ms_per_image"] / sp5["ms_per_image"]
        faster = "YOLO11s" if r < 1 else "YOLOv5s"
        P(f"  추론 : {faster} 가 {max(r, 1/r):.2f}배 빠름 "
          f"(11s {sp11['ms_per_image']:.2f}ms vs v5s {sp5['ms_per_image']:.2f}ms)")

    # ---------------------------------------------------------- 성능 비교
    sec("2. 정확도 비교 (test split, 동일 조건)")
    keys = ["mAP50", "mAP75", "mAP50_95", "AP_tiny", "AP_small", "AP_medium", "AP_large"]
    if pf11 or pf5:
        P(f"{'지표':<12}{'YOLO11s':>12}{'YOLOv5s':>12}{'차이':>12}")
        prows = []
        for k in keys:
            a, b = pf11.get(k), pf5.get(k)
            d = (a - b) if (a is not None and b is not None) else None
            P(f"{k:<12}{(f'{a:.4f}' if a is not None else '-'):>12}"
              f"{(f'{b:.4f}' if b is not None else '-'):>12}"
              f"{(f'{d:+.4f}' if d is not None else '-'):>12}")
            prows.append(dict(지표=k, YOLO11s=a, YOLOv5s=b, 차이=round(d, 4) if d else None))
        dump(pd.DataFrame(prows), "compare_performance.csv")
    else:
        P("  평가 결과가 없습니다. eval11 / eval5 를 먼저 실행하십시오.")

    # ---------------------------------------------------------- 클래스별 하락률
    sec("3. 클래스별 mAP50 -> mAP75 하락률")
    try:
        c11 = pd.read_csv(OUTD / f"eval_{N['v11']}" / "v11_by_class.csv")
        c5 = pd.read_csv(OUTD / f"eval_{N['v5']}" / "v5_by_class.csv")
        P(f"{'클래스':<14}{'11s':>10}{'v5s':>10}")
        comp = []
        for nm in NAMES:
            a = c11.loc[c11["클래스"] == nm, "하락률"].values
            b = c5.loc[c5["클래스"] == nm, "하락률"].values
            if len(a) and len(b):
                P(f"{nm:<14}{a[0]:>9.1f}%{b[0]:>9.1f}%")
                comp.append(dict(클래스=nm, YOLO11s_하락률=a[0], YOLOv5s_하락률=b[0]))
        if comp:
            dump(pd.DataFrame(comp), "compare_degradation.csv")
            P("\n>>> 두 모델 모두 Person 하락률 > Car 하락률 이면,")
            P("    anchor 유무를 바꿔도 같은 패턴 = 모델이 아니라 손실·데이터 차원의 문제")
    except Exception as e:
        P("  [skip] 클래스별 비교 불가:", type(e).__name__)

    # ---------------------------------------------------------- 통합표
    sec("4. 통합 요약")
    summary = []
    for key, tr, sp, pf in [("YOLO11s", tr11, sp11, pf11), ("YOLOv5s", tr5, sp5, pf5)]:
        summary.append({
            "모델": key,
            "학습시간(분)": round(tr["total_sec"] / 60, 2) if tr.get("total_sec") else None,
            "에폭당(초)": tr.get("sec_per_epoch"),
            "추론(ms/장)": sp.get("ms_per_image"),
            "FPS": sp.get("fps"),
            "mAP50": pf.get("mAP50"),
            "mAP75": pf.get("mAP75"),
            "mAP50-95": pf.get("mAP50_95"),
            "AP_small": pf.get("AP_small"),
        })
    sdf = pd.DataFrame(summary)
    P(sdf.to_string(index=False))
    dump(sdf, "compare_summary.csv")

    P(""); P("산출물:", out)
    write_log("compare_log.txt")
    record_timing("compare", time.perf_counter() - t0)


# ==============================================================================
# 진입점
# ==============================================================================

STAGES = {
    "build":   stage_build,
    "setup5":  stage_setup5,
    "train11": stage_train11,
    "eval11":  stage_eval11,
    "train5":  stage_train5,
    "eval5":   stage_eval5,
    "compare": stage_compare,
}

ALL_ORDER = ["build", "setup5", "train11", "eval11", "train5", "eval5", "compare"]


def main():
    ap = argparse.ArgumentParser(
        description="HIT-UAV : YOLO11s vs YOLOv5s 학습/추론 시간 + 성능 비교",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"],
                    help="실행할 단계 ('all' = build~compare 전부)")
    ap.add_argument("--smoke", action="store_true",
                    help="train11 / train5 를 3에폭만 실행 (동작 확인용)")
    ap.add_argument("--force", action="store_true",
                    help="build 단계에서 기존 dataset 을 무시하고 다시 생성")
    args = ap.parse_args()

    ok = lambda p: "[OK]  " if p.exists() else "[없음]"
    print("=" * 70)
    print(" HIT-UAV  YOLO11s vs YOLOv5s")
    print("=" * 70)
    print(" 작업 폴더 :", ROOT)
    print(" 장치      :", gpu_name())
    print("-" * 70)
    print(f" {ok(RAW)} raw          {RAW}")
    print(f" {ok(V5DIR)} yolov5-7.0   {V5DIR}")
    print(f" {ok(DSD)} dataset      {DSD}")
    print(f" {ok(RUNS)} runs         {RUNS}")
    print("=" * 70)

    # raw 가 없으면 build 계열은 어차피 실패하므로 먼저 안내
    if not RAW.exists() and args.stage in ("build", "all"):
        print()
        print(" HIT-UAV 원본이 필요합니다. 아래 위치에 넣어주십시오:")
        print("  ", RAW)
        print(" (폴더 안에 normal_xml / normal_json 이 있으면 됩니다)")
        print()

    # smoke 플래그를 받는 단계 (학습·평가·비교가 같은 이름을 써야 함)
    SMOKE_AWARE = ("train11", "eval11", "train5", "eval5", "compare")

    def run_stage(s):
        if s == "build":
            stage_build(force=args.force)
        elif s in SMOKE_AWARE:
            STAGES[s](smoke=args.smoke)
        else:
            STAGES[s]()

    t_all = time.perf_counter()
    if args.stage == "all":
        for s in ALL_ORDER:
            print(f"\n\n########## STAGE: {s}"
                  f"{'  (SMOKE)' if args.smoke else ''} ##########\n")
            run_stage(s)
        record_timing("ALL", time.perf_counter() - t_all,
                      f"전체 파이프라인 smoke={args.smoke}")
    else:
        run_stage(args.stage)

    print("\n완료. 시간 기록:", META / "timing_log.csv")


if __name__ == "__main__":
    main()
