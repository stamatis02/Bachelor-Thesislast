# Predicting Virality of YouTube Videos with Deep Learning Techniques

Bachelor's Thesis, Computer Science — **Stamatis Psaros**

An autonomous pipeline that discovers brand-new YouTube videos, tracks them for 3 days, and trains a PyTorch neural network to predict **from the first 8 hours** whether a video will go viral.

---

## How it works

Every 2 hours, one cycle runs through 6 stages:

| Stage | When | What it does |
|---|---|---|
| **1. Ingest** | 0–2 h | Discovers new non-Short videos that are already gaining views |
| **2. Track** | 8 h | Records views, likes and comments. The model features are computed here |
| **3. Track** | 24 h | Recorded for analysis only (**not** used by the model) |
| **4. Label** | 72 h | "The Teacher" assigns the ground truth: viral = 1 or 0 |
| **5. Train** | every cycle | "The Student" (neural network) is retrained from scratch on all labelled rows |
| **6. Predict** | every cycle | Scores every video that just reached 8 h, **before** its real label exists |

### Definition of viral

A video is labelled viral when, at 72 hours, **both** conditions hold:

- **Reach:** views ≥ 3 × channel subscribers. The video spread beyond its existing audience.
- **Genuine engagement:** (likes + comments) / views ≥ 0.3%. This filters out bot-inflated views.

> The engagement threshold was originally 2%. It was lowered to 0.3% because truly viral videos are watched mostly by non-subscribers, who rarely like or comment. At 2%, 30 of the 37 videos that hit the reach target were rejected.

### Model features (all known at the 8-hour mark)

| Feature | Meaning |
|---|---|
| `log_views_per_minute` | Velocity: views per minute at 8 h |
| `likes_per_view` | Likes / views |
| `comments_per_view` | Comments / views |
| `title_length` | Title length in characters |
| `tags_count` | Number of tags |
| `log_views_per_sub_8h` | Views at 8 h relative to channel subscribers |
| `log_subscribers` | Channel size |
| `log_acceleration` | Whether the video is speeding up or slowing down between 0–2 h and 8 h |
| `log_duration` | Video duration |

**Zero data leakage:** on every training run the code checks (`assert_no_leakage`) that no feature comes from the 24 h or 72 h observations. If one is ever added by mistake, the program stops.

### Model

- **Neural network (MLP):** 9 → 32 → 16 → 1, with BatchNorm, Dropout, weight decay and early stopping.
- **Class imbalance:** viral videos are rare (~8%), so the loss uses `pos_weight`.
- **Baselines:** every training run also evaluates a logistic regression with 5-fold cross-validation, to show whether the neural network adds value beyond a simple model.

---

## Repository contents

| File | Description |
|---|---|
| `youtube_virality_engine.py` | The entire pipeline in a single file |
| `youtube_predictive_datasetss.xlsx` | The dataset, updated automatically every cycle |
| `models/metrics.json` | Results of the latest training run |
| `models/metrics_history.jsonl` | History of results, for plotting progress over time |
| `models/virality_net.pt`, `preprocessor.joblib` | The trained model |
| `last_cycle.log` | Log of the most recent cycle |
| `.github/workflows/engine.yml` | GitHub Actions workflow for automated runs |
| `requirements.txt` | Python dependencies |

---

## Automation

- The pipeline runs on **GitHub Actions**, so no local machine needs to stay on.
- **cron-job.org** triggers the workflow every 2 hours through the GitHub API. GitHub's built-in `schedule` is not used: on free accounts it only fired every 6–8 hours.
- After each cycle, the updated dataset and model are committed back to the repo as `cycle <timestamp>`.
- The YouTube API key is stored as a GitHub Secret (`YOUTUBE_API_KEY`) and is never in the code.

---

## Running locally

```bash
pip install -r requirements.txt
export YOUTUBE_API_KEY="your_key_here"

python youtube_virality_engine.py --once            # single cycle
python youtube_virality_engine.py                   # run forever (every 2 hours)
python youtube_virality_engine.py --once --no-train # data collection only
python youtube_virality_engine.py --self-test       # offline logic check, no API needed
```

---

## Limitations

- **Sample:** videos from channels with 500–300,000 subscribers that already show traction in their first 2 hours. Not representative of all of YouTube.
- **YouTube API quota:** 10,000 units per day, which limits how many videos can be collected.
- **Time:** each video needs 3 days before it gets a label, so the dataset grows gradually.
