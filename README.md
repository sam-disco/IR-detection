# HIT-UAV : YOLO11s vs YOLOv5s — 학습/추론 시간 + 성능 비교

`hituav_bench.py` 한 파일입니다. 기존 9개 스크립트(step1c ~ step5)를 합치되,
그래프·데이터분석 단계를 걷어내고 **시간 측정**을 대폭 강화했습니다.

> 이전에 드린 `yolo_comparison.py`는 더미 데이터를 만드는 잘못된 버전입니다. 삭제하세요.

---

## 무엇을 남기고 무엇을 뺐나

| 원본 | 처리 |
|---|---|
| step1c_build | **유지** → `build` (HIT-UAV 원본을 그대로 불러옴) |
| step2_analysis | 삭제 (박스 분포 분석 + 그림 5장) |
| step2b_stratify | 삭제 (고도/각도 교란, IoU 민감도 + 그림 4장) |
| step3a_eval | **유지·축소** → `eval11` (조건별 AP 분해와 그림 제거) |
| step3c_control | 삭제 (교란 통제, 위치 오차) |
| step4a_patch_v5 | **유지** → `setup5` (+ yolov5 자동 clone 추가) |
| step4b_train_v5 | **유지** → `train5` |
| step4c_eval_v5 | **유지·축소** → `eval5` |
| step5_figures | 삭제 (하드코딩 값으로 그림 3장) |
| — | **추가** → `compare` (시간 + 성능 통합 비교표) |

matplotlib 의존성 자체가 사라졌습니다.

---

## 폴더 구성

**모든 것이 `IR data_new` 안에서 끝납니다.** 다른 폴더는 참조하지 않습니다.

```
C:\Users\User\Desktop\IR data_new\
├── hituav_bench.py
├── raw\            ← 직접 준비할 유일한 것 (HIT-UAV 원본)
├── yolov5-7.0\     ← setup5 가 자동으로 내려받음
├── dataset\        ← build 가 생성
├── meta\           ← 시간 기록 · COCO GT · 로그
├── outputs\        ← 평가 결과
└── runs\           ← 학습된 가중치
```

### 준비해야 할 것은 `raw` 하나뿐

기존 `IR data\raw` 폴더를 통째로 `IR data_new\raw` 로 **복사**하면 됩니다
(탐색기에서 Ctrl+C → Ctrl+V). 원본은 그대로 두는 편이 안전합니다.

`raw` 안의 구조는 HIT-UAV 공식 배포판 그대로면 됩니다:

```
raw\...\normal_xml\Annotations\*.xml       ← 박스 좌표 (VOC)
raw\...\normal_json\train|val|test\*.jpg   ← 원저자 분할 (2029/290/579)
```

하위 깊이는 상관없습니다. 코드가 `normal_xml` 폴더를 재귀로 찾습니다.

`yolov5-7.0` 은 옮길 필요 없습니다 — `setup5` 가 git clone 하고,
git 이 없으면 zip 으로 자동 전환합니다.

---

## 실행 순서

```powershell
cd "C:\Users\User\Desktop\IR data_new"

python hituav_bench.py build      # 원본 → YOLO 구조 + COCO GT (1~2분)
python hituav_bench.py setup5     # yolov5-7.0 확보 + 호환 패치
python hituav_bench.py train11    # YOLO11s 학습   ← 시간 측정
python hituav_bench.py eval11     # YOLO11s 추론   ← 시간 측정
python hituav_bench.py train5     # YOLOv5s 학습   ← 시간 측정
python hituav_bench.py eval5      # YOLOv5s 추론   ← 시간 측정
python hituav_bench.py compare    # 최종 비교표
```

한 번에:

```powershell
python hituav_bench.py all
python hituav_bench.py all --smoke   # 3에폭만 — 전체 흐름 점검용 (10분 내외)
```

**처음이면 `--smoke` 로 한 번 돌려보고 본 학습을 권합니다.**

---

## 시간을 어떻게 재는가

### 학습
- 전체 wall time (`time.perf_counter`)
- **YOLO11**: ultralytics 콜백(`on_train_epoch_start/end`)으로 **에폭별 실측**
  → `meta/epoch_times_yolo11.csv`
- **YOLOv5**: `results.csv` 행 수로 완료 에폭을 확인하고 총시간을 나눠 평균 산출
  (v5는 에폭 콜백 후크가 없어 평균값입니다)

### 추론
두 모델 모두 **동일 조건**으로 맞췄습니다.

- `conf=0.001`, `NMS IoU=0.7`, `max_det=300`, `imgsz=640`
- **batch=1** — 배치로 묶으면 처리량은 좋아지지만 장당 지연시간이 왜곡됩니다
- **워밍업 20장을 측정에서 제외** (첫 추론은 CUDA 커널 컴파일 때문에 느립니다)
- 모든 GPU 구간에 `torch.cuda.synchronize()` — 비동기 실행 때문에 시간이 과소측정되는 걸 막습니다

측정 결과:

| 항목 | 뜻 |
|---|---|
| `ms_per_image` | 장당 평균 (전처리+추론+NMS 전부 포함) |
| `ms_median` / `ms_p90` | 중앙값 / 상위 10% — 평균만 보면 튀는 값에 속습니다 |
| `fps` | 1000 / 평균 ms |
| `ms_preprocess` | 리사이즈·정규화·GPU 전송 |
| `ms_inference` | 순수 forward |
| `ms_postprocess` | NMS |

---

## 결과 파일

```
meta/
├── timing_log.csv              ★ 모든 단계 시간이 누적 (재실행해도 이어붙음)
├── epoch_times_yolo11.csv        에폭별 실측 시간
├── epoch_times_yolov5.csv
├── train_yolo11.json           / train_yolov5.json    학습 통계
├── speed_yolo11.json           / speed_yolov5.json    추론 속도 상세
├── perf_yolo11.json            / perf_yolov5.json     mAP
├── image_meta.csv                고도/각도/주야 메타
└── coco_gt_test.json             COCO 평가용 정답

outputs/
├── eval_yolo11s_base/  speed.csv, v11_overall.csv, v11_by_class.csv, detections_test.json
├── eval_yolov5s_base/  speed.csv, v5_overall.csv,  v5_by_class.csv,  detections_test.json
└── compare/
    ├── compare_summary.csv     ★ 이거 하나면 발표에 충분
    ├── compare_time.csv          시간 상세
    ├── compare_performance.csv   mAP 상세
    └── compare_degradation.csv   클래스별 mAP50→75 하락률
```

### `timing_log.csv` 예시

```csv
timestamp,stage,seconds,hh:mm:ss,device,note,detail
2026-09-01T14:02:11,build,73.4,00:01:13,NVIDIA RTX 4070,2898 images,
2026-09-01T15:41:55,train11,5844.2,01:37:24,NVIDIA RTX 4070,100ep batch16,"{""epochs_done"": 100, ""sec_per_epoch"": 58.44}"
2026-09-01T15:44:03,eval11,127.8,00:02:07,NVIDIA RTX 4070,"579 images, 173400 dets","{""ms_per_image"": 8.62, ""fps"": 116.0}"
```

### `compare_summary.csv` 예시

```csv
모델,학습시간(분),에폭당(초),추론(ms/장),FPS,mAP50,mAP75,mAP50-95,AP_small
YOLO11s,97.4,58.44,8.62,116.0,0.8412,0.5231,0.4987,0.4432
YOLOv5s,112.8,67.68,9.31,107.4,0.8305,0.5012,0.4813,0.4731
```

---

## 설치

```powershell
pip install ultralytics pycocotools pandas numpy
pip install "setuptools<81"          # YOLOv5 v7.0 이 pkg_resources 를 씁니다
```

GPU (CUDA 12.1 기준):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`setup5` 가 yolov5-7.0 을 자동으로 확보합니다 (git → zip 순서로 시도).
둘 다 막히면 수동으로:

```powershell
git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5.git "C:\Users\User\Desktop\IR data_new\yolov5-7.0"
```

또는 [v7.0.zip](https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip) 을 받아
`IR data_new` 에 풀고 폴더명을 `yolov5-7.0` 으로 맞추면 됩니다.

> 기존 `IR data\venv_yolov5` 가상환경을 쓰고 계셨다면, 그 환경을 그대로 활성화한 채
> 새 폴더에서 실행해도 됩니다. 가상환경 위치는 코드와 무관합니다.

---

## 자주 걸리는 지점

**`raw 폴더 없음` / `normal_xml 폴더를 찾을 수 없습니다`**
→ `IR data_new\raw` 아래 어딘가에 `normal_xml` 폴더가 있어야 합니다.
기존 `IR data\raw` 를 복사해 오세요 (하위 깊이는 상관없음).

**`분할 수가 (2029, 290, 579) 와 다릅니다`**
→ 경고만 뜨고 진행됩니다. HIT-UAV 버전이 다를 수 있습니다.

**CUDA out of memory**
→ 코드 상단 `V11_BATCH` / `V5_BATCH` 를 8 또는 4 로 낮추세요.

**YOLOv5 에서 `torch.load` / `np.trapz` 에러**
→ `apply_v5_adapters()` 가 자동 처리합니다. 그래도 나면 torch 버전을 알려주세요.

**시간 비교가 공정한가?**
→ 두 모델 모두 같은 GPU, 같은 batch, 같은 imgsz, 같은 conf/NMS, 워밍업 제외,
`synchronize()` 적용입니다. 다만 **다른 시점에 측정하면 GPU 온도·클럭 때문에
5~10% 차이가 날 수 있으니**, 정밀 비교가 필요하면 `eval11`과 `eval5`를 연달아 돌리세요.

---

## 시간을 줄이고 싶다면

코드 상단에서:

```python
V11_EPOCHS = 100   # → 30 정도로도 경향은 보입니다
V5_EPOCHS  = 100
V11_BATCH  = 16    # GPU 여유 있으면 32 (단, 두 모델을 같은 값으로!)
V5_BATCH   = 16
```

배치 크기는 **두 모델을 반드시 같은 값**으로 두어야 비교가 성립합니다.
