"""
End-to-end verification of youtube_virality_engine.py against a MOCK YouTube API.

Runs offline — no API key, no quota, no network. It drives a fake clock through
t+0h, t+8h, t+24h and t+72h so every stage, every filter and the full run_cycle
orchestration are exercised in seconds instead of three days.

    python test_pipeline_mock.py        # exit code 0 = everything passed
"""
import sys, types, os
from datetime import datetime, timedelta, timezone

import numpy as np, pandas as pd
import youtube_virality_engine as E

FAILS = []
def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond: FAILS.append(name)

# ---- controllable clock ----------------------------------------------------
CLOCK = {"now": datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)}
E.utcnow = lambda: CLOCK["now"]

T0 = CLOCK["now"]

# ---- mock API --------------------------------------------------------------
def mk_video(vid, minutes_old, dur, views, likes, comments, ch, tags=5, title="A title"):
    return {
        "id": vid,
        "snippet": {"title": title, "channelId": ch, "channelTitle": ch,
                    "categoryId": "22", "tags": ["t"] * tags, "description": "d" * 40,
                    "publishedAt": E.to_iso(T0 - timedelta(minutes=minutes_old))},
        "statistics": {"viewCount": str(views), "likeCount": str(likes),
                       "commentCount": str(comments)},
        "contentDetails": {"duration": dur},
    }

CATALOG = {
    "GOOD1": mk_video("GOOD1", 70, "PT10M", 1200, 60, 10, "CH_BIG"),
    "GOOD2": mk_video("GOOD2", 95, "PT3M30S", 400, 12, 3, "CH_SMALLOK"),
    "SHORT1": mk_video("SHORT1", 30, "PT45S", 9000, 800, 40, "CH_BIG"),      # Short
    "LOWV": mk_video("LOWV", 60, "PT12M", 120, 5, 1, "CH_BIG"),              # <300 views
    "LOWSUB": mk_video("LOWSUB", 60, "PT12M", 5000, 300, 20, "CH_TINY"),     # <500 subs
    "OLD": mk_video("OLD", 400, "PT12M", 5000, 300, 20, "CH_BIG"),           # too old
    "HIDDEN": mk_video("HIDDEN", 60, "PT12M", 900, 40, 5, "CH_HIDDEN"),      # hidden subs
    "EXACT180": mk_video("EXACT180", 60, "PT3M", 900, 40, 5, "CH_BIG"),      # exactly 180s
}
CHANNELS = {
    "CH_BIG": {"id": "CH_BIG", "statistics": {"subscriberCount": "10000", "videoCount": "300"}},
    "CH_SMALLOK": {"id": "CH_SMALLOK", "statistics": {"subscriberCount": "500", "videoCount": "10"}},
    "CH_TINY": {"id": "CH_TINY", "statistics": {"subscriberCount": "120", "videoCount": "4"}},
    "CH_HIDDEN": {"id": "CH_HIDDEN", "statistics": {"hiddenSubscriberCount": True}},
}

class FakeClient:
    """Mimics YouTubeClient. STATS drives what later checkpoints return."""
    def __init__(self):
        self.batch_calls = 0
        self.stats = {}          # vid -> (views, likes, comments)
        self.deleted = set()
    def search_recent_video_ids(self, after, before, query, max_results=50):
        return list(CATALOG.keys())
    def get_videos(self, ids, parts="snippet,statistics,contentDetails"):
        self.batch_calls += 1
        out = {}
        for vid in ids:
            if vid in self.deleted or vid not in CATALOG:
                continue
            item = {k: dict(v) if isinstance(v, dict) else v for k, v in CATALOG[vid].items()}
            if vid in self.stats:
                v, l, c = self.stats[vid]
                item["statistics"] = {"viewCount": str(v), "likeCount": str(l),
                                      "commentCount": str(c)}
            out[vid] = item
        return out
    def get_channels(self, ids):
        return {cid: CHANNELS[cid] for cid in dict.fromkeys(ids) if cid in CHANNELS}

client = FakeClient()
store = E.ExcelStore(E.Path("mock_dataset.xlsx"))
if store.path.exists(): store.path.unlink()

print("\n== STAGE 1: ingestion filters ==")
df = store.load()
df = E.ingest_new_videos(client, df)
df = store.save(df)
kept = sorted(df["video_id"].tolist())
check("keeps only GOOD1+GOOD2", kept == ["GOOD1", "GOOD2"])
check("Shorts excluded", "SHORT1" not in kept)
check("exactly-180s excluded (strict >)", "EXACT180" not in kept)
check("sub-300-view excluded", "LOWV" not in kept)
check("sub-500-sub channel excluded", "LOWSUB" not in kept)
check("hidden-subscriber channel excluded", "HIDDEN" not in kept)
check("out-of-window video excluded", "OLD" not in kept)
check("all rows PENDING", (df["is_viral"] == E.PENDING).all())
check("subscribers captured", df.set_index("video_id").at["GOOD1", "subscribers"] == 10000)
check("title_length/tags_count set",
      df.set_index("video_id").at["GOOD1", "tags_count"] == 5
      and df.set_index("video_id").at["GOOD1", "title_length"] == len("A title"))

print("\n== idempotency: a second ingest must not duplicate ==")
df = E.ingest_new_videos(client, df); df = store.save(df)
check("no duplicate rows", len(df) == 2)

print("\n== STAGE 2: 8h checkpoint + feature engineering ==")
CLOCK["now"] = T0 + timedelta(hours=8)     # GOOD1 is 70min older than T0
client.stats = {"GOOD1": (9600, 480, 96), "GOOD2": (1000, 20, 4)}
before = client.batch_calls
df = E.track_checkpoint(client, df, 8, E.utcnow()); df = store.save(df)
row = df.set_index("video_id").loc["GOOD1"]
age = row["age_minutes_8h"]
check("one batched request for both videos", client.batch_calls - before == 1)
check("stage_8h_status OK", row["stage_8h_status"] == E.STATUS_OK)
check("views_8h stored", row["views_8h"] == 9600)
check("age uses real published_at (~551 min)", abs(age - (8 * 60 + 70)) < 1.0)
check("views_per_minute = views/measured age", abs(row["views_per_minute"] - 9600 / age) < 1e-6)
check("likes_per_view correct", abs(row["likes_per_view"] - 480 / 9600) < 1e-9)
check("comments_per_view correct", abs(row["comments_per_view"] - 96 / 9600) < 1e-9)
check("24h/72h still DUE", row["stage_24h_status"] == E.STATUS_DUE
      and row["stage_72h_status"] == E.STATUS_DUE)
check("8h not re-run", E.find_due_rows(df, 8, E.utcnow()) == [])

print("\n== STAGE 3: 24h checkpoint ==")
CLOCK["now"] = T0 + timedelta(hours=24, minutes=30)
client.stats = {"GOOD1": (50000, 2400, 500), "GOOD2": (1500, 30, 6)}
df = E.track_checkpoint(client, df, 24, E.utcnow()); df = store.save(df)
row = df.set_index("video_id").loc["GOOD1"]
check("views_24h stored", row["views_24h"] == 50000)
check("8h features untouched by the 24h pass",
      abs(row["views_per_minute"] - 9600 / age) < 1e-6)

print("\n== STAGE 4: 72h ground truth ==")
CLOCK["now"] = T0 + timedelta(hours=72, minutes=30)
# GOOD1: 100k views vs 10k subs (>=3x) and (5000+1000)/100000 = 6% -> VIRAL
# GOOD2: 1600 views vs 500 subs (>=3x) but (3+1)/1600 = 0.25% < 0.3% -> NOT viral
client.stats = {"GOOD1": (100000, 5000, 1000), "GOOD2": (1600, 3, 1)}
df = E.label_ground_truth(client, df, E.utcnow()); df = store.save(df)
idx = df.set_index("video_id")
check("GOOD1 labelled 1 (reach + resonance)", int(idx.at["GOOD1", "is_viral"]) == 1)
check("GOOD2 labelled 0 (reach but no resonance)", int(idx.at["GOOD2", "is_viral"]) == 0)
check("engagement_ratio computed", abs(idx.at["GOOD1", "engagement_ratio"] - 0.06) < 1e-9)
check("views_to_subs_ratio computed", abs(idx.at["GOOD1", "views_to_subs_ratio"] - 10.0) < 1e-9)
check("row_status LABELLED", (df["row_status"] == "LABELLED").all())

print("\n== deleted-video handling ==")
client.deleted = {"GOOD1"}
extra = pd.DataFrame([{ "video_id": "GOOD1b", "published_at": E.to_iso(T0 - timedelta(hours=9)),
                        "channel_id": "CH_BIG", "is_viral": E.PENDING,
                        "stage_8h_status": E.STATUS_DUE, "stage_24h_status": E.STATUS_DUE,
                        "stage_72h_status": E.STATUS_DUE }])
d2 = E.ExcelStore._normalise(pd.concat([df, extra], ignore_index=True))
d2 = E.track_checkpoint(client, d2, 8, E.utcnow())
check("missing-from-API row is not crashed on", len(d2) == 3)

print("\n== Excel round-trip fidelity ==")
reloaded = store.load()
check("row count preserved", len(reloaded) == 2)
check("schema preserved", list(reloaded.columns) == E.SCHEMA)
check("labels preserved as numbers",
      sorted(pd.to_numeric(reloaded["is_viral"], errors="coerce").tolist()) == [0, 1])
check("timestamps re-parse", E.from_iso(reloaded.at[0, "checked_at_8h"]) is not None)

print("\n== training frame / leakage contract ==")
frame = E.build_training_frame(reloaded)
check("training frame holds only the model features + target",
      list(frame.columns) == E.FEATURE_COLUMNS + [E.TARGET_COLUMN])
check("no 24h/72h column can reach the model",
      all(not any(p in c for p in E.LEAKAGE_PATTERNS) for c in E.FEATURE_COLUMNS))
try:
    E.assert_no_leakage(["views_per_minute", "views_24h"]); leak_ok = False
except RuntimeError: leak_ok = True
check("guard rejects a 24h feature", leak_ok)

print("\n== misc unit checks ==")
check("PT4M13S -> 253", E.parse_iso_duration("PT4M13S") == 253)
check("PT1H2M3S -> 3723", E.parse_iso_duration("PT1H2M3S") == 3723)
check("P1DT2H -> 93600", E.parse_iso_duration("P1DT2H") == 93600)
check("live/garbage -> 0", E.parse_iso_duration("P0D") == 0)
check("safe_div by zero -> NaN", np.isnan(E.safe_div(5, 0)))
check("chunked(120, 50) -> 3 batches", len(list(E.chunked(list(range(120)), 50))) == 3)
q = E.QuotaGuard(950)
q.charge("search", 9)   # 900 units spent
check("quota blocks the 11th search", q.can_afford("search") is False)
check("quota still allows cheap batch calls", q.can_afford("videos") is True)

sched = pd.DataFrame([
    {"video_id": "A", "published_at": E.to_iso(E.utcnow() - timedelta(hours=3)), "stage_8h_status": E.STATUS_DUE},
    {"video_id": "B", "published_at": E.to_iso(E.utcnow() - timedelta(hours=9)), "stage_8h_status": E.STATUS_DUE},
    {"video_id": "C", "published_at": E.to_iso(E.utcnow() - timedelta(hours=40)), "stage_8h_status": E.STATUS_DUE},
    {"video_id": "D", "published_at": E.to_iso(E.utcnow() - timedelta(hours=9)), "stage_8h_status": E.STATUS_OK},
])
due = E.find_due_rows(sched, 8, E.utcnow())
obs, missed = E.expire_missed(sched, 8, due, E.utcnow())
check("scheduler picks aged+unobserved only", sorted(due) == [1, 2])
check("grace window keeps B, expires C",
      obs == [1] and missed == 1 and sched.at[2, "stage_8h_status"] == E.STATUS_MISSED)


print("\n== full run_cycle orchestration (ingest -> track -> label -> train -> predict) ==")
rng = np.random.default_rng(7)
seed_rows = []
for i in range(220):
    vpm = float(abs(rng.normal(8, 6))); lpv = float(abs(rng.normal(0.03, 0.02)))
    cpv = float(abs(rng.normal(0.004, 0.003)))
    lbl = 1 if 0.45*vpm + 60*lpv + 200*cpv + rng.normal(0, 1.5) > 6.5 else 0
    seed_rows.append({"video_id": f"S{i:04d}", "published_at": E.to_iso(E.utcnow()-timedelta(hours=200)),
                      "subscribers": 5000, "views_8h": int(vpm*480), "likes_8h": 40, "comments_8h": 4,
                      "age_minutes_8h": 480.0, "views_per_minute": vpm, "likes_per_view": lpv,
                      "comments_per_view": cpv, "title_length": 50, "tags_count": 9,
                      "stage_8h_status": E.STATUS_OK, "stage_24h_status": E.STATUS_OK,
                      "stage_72h_status": E.STATUS_OK, "is_viral": lbl, "row_status": "LABELLED"})
seeded = E.ExcelStore._normalise(pd.DataFrame(seed_rows))
store2 = E.ExcelStore(E.Path("cycle_dataset.xlsx")); store2.save(seeded)
CLOCK["now"] = T0
client2 = FakeClient()
quota = E.QuotaGuard(10000)
E.run_cycle(client2, store2, quota, do_ingest=True, do_train=True)
CLOCK["now"] = T0 + timedelta(hours=8, minutes=30)
E.run_cycle(client2, store2, quota, do_ingest=False, do_train=True)
final = store2.load()
newly = final[final["video_id"].isin(["GOOD1", "GOOD2"])]
check("run_cycle ingested + tracked the new videos", len(newly) == 2)
check("8h features exist after cycle 2", newly["views_per_minute"].notna().all())
probs = pd.to_numeric(newly["pred_prob"], errors="coerce")
check("run_cycle produced predictions for the 8h cohort", probs.notna().all())
check("predictions are probabilities", bool(((probs >= 0) & (probs <= 1)).all()))
check("pred_label matches the 0.5 threshold",
      bool((pd.to_numeric(newly["pred_label"]) == (probs >= 0.5).astype(int)).all()))
check("model version stamped on predictions", newly["pred_model_version"].notna().all())
check("seeded labelled rows untouched by inference",
      bool(pd.to_numeric(final[final["video_id"].astype(str).str.startswith("S")]["pred_prob"],
                         errors="coerce").isna().all()))
try: E.Path("cycle_dataset.xlsx").unlink()
except OSError: pass

for f in ["mock_dataset.xlsx", "virality_engine.log"]:
    try: os.unlink(f)
    except OSError: pass

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
