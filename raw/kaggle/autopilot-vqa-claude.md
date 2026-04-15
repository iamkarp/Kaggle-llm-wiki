# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

**VQA - AUTOPILOT CVPR** — A Kaggle competition for incident-centric Visual Question Answering (VQA) from dashcam video, supporting the AUTOPILOT CVPR workshop.

- Competition URL: `https://kaggle.com/competitions/AUTOPILOT-VQA`
- Contact: rrabinow@uccs.edu, aalshami@uccs.edu
- Evaluation metric: **mean per-question accuracy** across all VQA fields
- Prize: $300 for 1st place

## Task

Multi-output classification: for each `video_id`, predict one integer label per question column. All answers must be numeric per the legend below. Unknown answers use `-1`. Non-applicable answers (e.g., vehicle type when entity is not a vehicle) use `-2` and are not scored.

## Data

- Dataset: 2COOOL benchmark (from COOOL, DADA-2000, and Nexar collision footage)
- Video files + heatmaps: Google Drive folder `1JYhSPv0zQQ_EBx2G4ljyF6CNXlxkRNVB`
- `autopilot` file: Google Drive HTML page for `2COOOL_Competition_Video_Data_Final`
- `readme.txt`: Full competition description and label legend from Kaggle

## Submission Format

CSV with one row per `video_id`, one column per question (exact header from `sample_submission.csv`). Column names contain newlines — copy them directly from the sample file.

```
video_id,"A)Weather:\nQ1.a: ...","A)Weather:\nQ1.b: ...",...
269,0,1,2,...
```

## Label Legend (summary)

| Section | Questions | Key values |
|---------|-----------|------------|
| A) Weather | Q1.a (time of day), Q1.b (weather) | Day=0, Dawn=1, Dusk=2, Night=3 / Sunny=0, Cloudy=1, Rainy=2, ... |
| B) Light | Q2 | Daylight=0, Headlights only=1, Sunrise/Sunset=2, Streetlights=4 |
| C) Traffic Env | Q3.a (location), Q3.b (facilities) | Urban=0, Suburban=2, Rural=3 |
| D) Road Config | Q4.a (type), Q4.b (lanes), Q4.c (ego-lane dir) | Highway=1, Intersection=2, T-junction=3, ... |
| E) Incident | Q5.a (category), Q5.b (primary entity), Q5.c (primary vehicle type), Q5.e (primary behavior), Q5.f (secondary entity), Q5.g (secondary vehicle type), Q5.i (secondary behavior), Q5.j (incident peak) | |
| F) Prevention | Q6.a (primary prevention), Q6.b (secondary prevention) | |
| G) Impact | Q7.a–Q7.d (impact location on primary/secondary, side hit) | |
| H) Traffic Control | Q8 | None=1, Traffic light=3, Stop sign=10, ... |
| I) Road Surface | Q9.a (condition), Q9.b (material) | Dry=0, Wet=1, Snow/Ice=2 / Asphalt=0, Concrete=1, Gravel=3 |

Full numeric codes for every value are in `readme.txt` lines 134–388.

## Recommended Approach

Use Vision Language Models (VLMs) to process dashcam video frames and answer the structured question set. The competition encourages VLM-based solutions but allows any architecture.
