# AUTOPILOT VQA — Winning Strategy

## Competition Summary

- **Task**: 25-column multi-output classification from dashcam video
- **Metric**: Mean per-question accuracy across all 25 questions
- **Deadline**: April 15, 2026 | **Prize**: $300
- **Status**: 0 submissions so far — first-mover advantage is real

---

## Core Approach: Hierarchical VLM Pipeline + Ensemble

Based on the 2nd-place solution from the 2COOOL CVPR challenge (arXiv:2510.12190), which is the most directly applicable published result.

### Stack

| Model | Role | Hosting |
|-------|------|---------|
| **Qwen2.5-VL-32B** | Bulk inference, video understanding | vLLM on local RTX 3090/4090 |
| **Claude claude-sonnet-4-6** (`claude-sonnet-4-6`) | Causal reasoning (Q5, Q6), ensemble | Anthropic API |

Ensemble: majority vote per column; tie → trust Claude.

---

## Pipeline (3 Stages per Video)

### Stage 1 — Scene Captioning
Sample 5 keyframes using optical flow (motion boundaries, not uniform). For each frame:
```
"Describe this dashcam frame: weather, lighting, road type, visible entities, any incidents."
```

### Stage 2 — Incident Frame Detection
```
"Given frame descriptions: {captions}
Which 1-2 frames contain the incident peak? Return frame indices."
```

### Stage 3 — Structured Q&A on Incident Frames
Embed the full label legend in the prompt. Ask all 25 questions at once:
```
You are a traffic safety analyst reviewing dashcam footage.
Scene context: {stage1_captions}

Answer using ONLY the integer codes below. Unknown = -1. Not applicable = -2.
{full_legend}

Return JSON: {"Q1a": int, "Q1b": int, ..., "Q9b": int}
```

---

## Frame Extraction

```python
import cv2

def extract_keyframes(video_path, n=5):
    cap = cv2.VideoCapture(video_path)
    frames, diffs = [], []
    prev = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(gray, prev).mean()
            diffs.append((diff, frame))
        prev = gray
    cap.release()
    # Pick top-N highest-motion frames
    diffs.sort(key=lambda x: -x[0])
    return [f for _, f in diffs[:n]]
```

Key insight: **4-6 high-motion frames beat 64 uniform frames** — verified across multiple video VQA benchmarks.

---

## Claude API Integration

```python
import anthropic, base64, cv2, json

client = anthropic.Anthropic()

def frame_to_b64(frame) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

def predict_with_claude(frames, scene_context: str, legend: str) -> dict:
    content = []
    for frame in frames:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_to_b64(frame)}
        })
    content.append({"type": "text", "text": build_prompt(scene_context, legend)})

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}]
    )
    return json.loads(msg.content[0].text)
```

---

## N/A Auto-Fill Rules (Critical for Score)

Pre-code these deterministic rules before submission — wrong `-2` vs `-1` will hurt accuracy:

```python
def apply_na_rules(pred: dict) -> dict:
    # Vehicle type questions: -2 if entity is not a vehicle
    NON_VEHICLE_PRIMARY = {0, 2, 3, 10, 13, 29}   # pedestrian, animal, etc.
    NON_VEHICLE_SECONDARY = {0, 3, 4, 6}            # cyclist, object, pedestrian, animal

    if pred["Q5b"] in NON_VEHICLE_PRIMARY:
        pred["Q5c"] = -2   # primary vehicle type N/A

    if pred["Q5f"] in NON_VEHICLE_SECONDARY:
        pred["Q5g"] = -2   # secondary vehicle type N/A

    # Impact point questions: -2 if no collision
    if pred["Q5j"] == 1:   # no collision
        pred["Q7a"] = pred["Q7b"] = pred["Q7c"] = pred["Q7d"] = -2

    return pred
```

---

## Per-Question Strategy

| Category | Questions | Key frames to use |
|----------|-----------|-------------------|
| Weather / time | Q1.a, Q1.b | Any clear frame |
| Lighting | Q2 | Brightest / most representative frame |
| Traffic environment | Q3.a, Q3.b | Wide-view frame |
| Road configuration | Q4.a–Q4.c | Pre-incident frame (before things move) |
| **Incident entities/behavior** | **Q5.a–Q5.j** | **Peak frame + 1 before/after — use Claude** |
| **Prevention** | **Q6.a, Q6.b** | **Full sequence context — use Claude** |
| Impact point | Q7.a–Q7.d | Peak impact frame |
| Traffic control | Q8 | Wide-view frame (look for signs/signals) |
| Road surface | Q9.a, Q9.b | Ground-level frame |

---

## Submission Format

Column headers contain literal newlines — **copy them verbatim from `sample_submission.csv`**, do not type them by hand.

```python
import pandas as pd

sample = pd.read_csv("sample_submission.csv")
output = pd.DataFrame(predictions)  # same columns as sample
output.columns = sample.columns     # ensures exact header match
output.to_csv("submission.csv", index=False)
```

---

## Execution Timeline

| Days | Phase | Goal |
|------|-------|------|
| 1–3 | Data + baseline | Download videos, extract frames, single-model zero-shot submission |
| 4–8 | 3-stage pipeline | Hierarchical scene → incident → Q&A |
| 9–12 | Ensemble | Add Claude API, majority vote, N/A rules |
| 13–16 | Calibration | Error analysis per column, prompt refinement on weak questions |
| 17+ | Final push | Polish submission, maximize daily 5 submissions |

---

## Key Dependencies

```bash
pip install transformers accelerate qwen-vl-utils opencv-python pandas tqdm
pip install anthropic          # Claude API
pip install vllm               # local high-throughput Qwen inference
pip install gdown              # Google Drive download
```

---

## References

- **arXiv:2510.12190** — Hierarchical Reasoning with VLMs for Dashcam Incident Reports (2COOOL 2nd place) — blueprint for this pipeline
- **arXiv:2502.13923** — Qwen2.5-VL Technical Report
- **arXiv:2504.14526** — DVBench: Safety-Critical Driving Video Understanding (25-ability benchmark, directly aligned with this competition)
- **arXiv:2503.03848** — Nexar Dashcam Collision Prediction (CVPR 2025)
