"""
Shill Bidding Review Assistant  -  Streamlit prototype
=====================================================

Workflow
--------
  Business data / user input  ->  model output  ->  interpretation  ->  user action  ->  outcome + recommendation

The model is the team's selected notebook model:
  * Random Forest classifier, 9 behavioural features, 100 trees
  * Tuned settings from the auction-grouped GridSearchCV (max_depth=None, min_samples_leaf=1)
  * Decision cut-offs from the validation threshold study (0.7 = Escalate, 0.3 = Monitor)
  * Same auction-level train / validation / test split as the notebook (random_state=0)

Run locally
-----------
  pip install streamlit scikit-learn pandas numpy
  streamlit run app.py

Put "Shill Bidding Dataset.csv" in the same folder as this file
(or upload it from the sidebar).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Configuration (all values come from the notebook)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Shill Bidding Review Assistant", page_icon="🔎", layout="wide")

DATA_FILE = "Shill Bidding Dataset.csv"

FEATURES = [
    "Bidder_Tendency", "Bidding_Ratio", "Successive_Outbidding", "Last_Bidding",
    "Auction_Bids", "Starting_Price_Average", "Early_Bidding", "Winning_Ratio",
    "Auction_Duration",
]
ID_COLS = ["Record_ID", "Auction_ID", "Bidder_ID"]
DURATION_CHOICES = [1, 3, 5, 7, 10]  # observed auction lengths in days

# Final model settings chosen in the notebook (Sections 19A and 20)
RF_PARAMS = dict(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=0)
ESCALATE_THRESHOLD = 0.7   # best validation F1 (precision 1.000, recall 0.943)
MONITOR_THRESHOLD = 0.3    # high-recall setting (recall 0.991 on validation)

FEATURE_HELP = {
    "Bidder_Tendency": "Share of the bidder's activity concentrated on few sellers.",
    "Bidding_Ratio": "How often the bidder participates relative to the auction's bids.",
    "Successive_Outbidding": "Bidder repeatedly outbids themself while already winning.",
    "Last_Bidding": "Inactivity in the final stage of the auction.",
    "Auction_Bids": "Auction's bid count relative to concurrent auctions.",
    "Starting_Price_Average": "Starting-price indicator (not a raw dollar price).",
    "Early_Bidding": "Tendency to bid in the first quarter of the auction.",
    "Winning_Ratio": "Share of the bidder's auctions that they win.",
    "Auction_Duration": "Length of the auction in days.",
}


def pretty(name):
    return name.replace("_", " ")


def tier_of(score):
    if score >= ESCALATE_THRESHOLD:
        return "Escalate"
    if score >= MONITOR_THRESHOLD:
        return "Monitor"
    return "Clear"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_from_path(path):
    return pd.read_csv(path)


def load_training_data():
    """Use the CSV next to app.py if present, otherwise the sidebar upload."""
    local = Path(__file__).parent / DATA_FILE
    if local.exists():
        return read_csv_from_path(str(local)), f"Loaded `{DATA_FILE}` from the app folder."
    uploaded = st.sidebar.file_uploader(
        "Upload 'Shill Bidding Dataset.csv'", type="csv", key="training_upload"
    )
    if uploaded is not None:
        return pd.read_csv(uploaded), "Loaded uploaded training file."
    return None, None


# ----------------------------------------------------------------------------
# Model training (mirrors the notebook; cached so it runs once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the Random Forest from the notebook settings...")
def build_artifacts(df):
    # Auction-level split: all rows from one auction stay together (Notebook Section 5)
    auction_ids = df["Auction_ID"].drop_duplicates()
    development_ids, test_ids = train_test_split(auction_ids, test_size=0.2, random_state=0)
    train_ids, validation_ids = train_test_split(development_ids, test_size=0.2, random_state=0)

    train_df = df[df["Auction_ID"].isin(train_ids)]
    validation_df = df[df["Auction_ID"].isin(validation_ids)]
    test_df = df[df["Auction_ID"].isin(test_ids)]

    X_train, y_train = train_df[FEATURES], train_df["Class"]
    X_val, y_val = validation_df[FEATURES], validation_df["Class"]
    X_test, y_test = test_df[FEATURES], test_df["Class"]

    forest = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)

    def score_block(X, y, threshold):
        p = forest.predict_proba(X)[:, 1]
        pred = (p >= threshold).astype(int)
        return {
            "Precision": metrics.precision_score(y, pred, zero_division=0),
            "Recall": metrics.recall_score(y, pred, zero_division=0),
            "F1": metrics.f1_score(y, pred, zero_division=0),
            "ROC AUC": metrics.roc_auc_score(y, p),
            "Flagged records": int(pred.sum()),
            "Records": len(y),
        }

    val_rows = []
    for t in [0.3, 0.5, 0.7]:
        row = score_block(X_val, y_val, t)
        row["Threshold"] = t
        val_rows.append(row)
    validation_thresholds = pd.DataFrame(val_rows).set_index("Threshold")
    test_metrics = score_block(X_test, y_test, ESCALATE_THRESHOLD)
    test_metrics_monitor = score_block(X_test, y_test, MONITOR_THRESHOLD)

    # Sensitivity check: same model family without Successive_Outbidding (Notebook Section 22)
    f8 = [f for f in FEATURES if f != "Successive_Outbidding"]
    forest8 = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=5, random_state=0
    ).fit(train_df[f8], y_train)
    p8 = forest8.predict_proba(test_df[f8])[:, 1]
    pred8 = (p8 >= 0.3).astype(int)
    eight_feature_test = {
        "Precision": metrics.precision_score(y_test, pred8, zero_division=0),
        "Recall": metrics.recall_score(y_test, pred8, zero_division=0),
        "F1": metrics.f1_score(y_test, pred8, zero_division=0),
        "ROC AUC": metrics.roc_auc_score(y_test, p8),
    }

    # Auction-duration experiment (Notebook Section 23)
    dur_feats = ["Bidder_Tendency", "Bidding_Ratio", "Successive_Outbidding",
                 "Auction_Bids", "Starting_Price_Average", "Winning_Ratio"]
    per_auction = df.groupby("Auction_ID")[dur_feats + ["Auction_Duration"]].mean()
    d_train, d_val, d_test = (per_auction.loc[train_ids], per_auction.loc[validation_ids],
                              per_auction.loc[test_ids])
    lin = LinearRegression().fit(d_train[dur_feats], d_train["Auction_Duration"])
    base = d_train["Auction_Duration"].mean()
    duration_rows = []
    for split_name, part in [("Validation", d_val), ("Test", d_test)]:
        truth = part["Auction_Duration"]
        duration_rows.append({
            "Split": split_name,
            "Baseline MAE (days)": metrics.mean_absolute_error(truth, np.full(len(truth), base)),
            "Linear Regression MAE (days)": metrics.mean_absolute_error(truth, lin.predict(part[dur_feats])),
        })
    duration_results = pd.DataFrame(duration_rows).set_index("Split")

    return {
        "forest": forest,
        "train_df": train_df,
        "test_df": test_df,
        "normal_median": X_train[y_train == 0].median(),
        "suspicious_median": X_train[y_train == 1].median(),
        "base_rate": float(df["Class"].mean()),
        "importance": pd.Series(forest.feature_importances_, index=FEATURES).sort_values(),
        "validation_thresholds": validation_thresholds,
        "test_metrics": test_metrics,
        "test_metrics_monitor": test_metrics_monitor,
        "eight_feature_test": eight_feature_test,
        "duration_results": duration_results,
    }


# ----------------------------------------------------------------------------
# Interpretation helpers
# ----------------------------------------------------------------------------
def driver_table(model, X, reference):
    """
    Per-record 'what pushes this score up' table.
    For each feature, replace it with the typical normal-bidder value and measure how much the
    risk score falls. A big drop means that feature is a main reason for the score.
    """
    base = model.predict_proba(X)[:, 1]
    drops = {}
    for f in FEATURES:
        altered = X.copy()
        altered[f] = reference[f]
        drops[f] = base - model.predict_proba(altered)[:, 1]
    return pd.DataFrame(drops, index=X.index)


def top_driver_label(drops_row):
    best = drops_row.idxmax()
    return pretty(best) if drops_row[best] > 0.01 else "—"


def next_step_text(tier):
    return {
        "Escalate": "Open a manual investigation: pull the bidder's bid history in this auction, check "
                    "whether the same few sellers recur, and place the auction on seller review. "
                    "Do not penalise automatically — scores are leads, not proof.",
        "Monitor": "Add to the watchlist. Re-score after the next few auctions and escalate if the "
                   "score rises or the same bidder appears again in another flagged auction.",
        "Clear": "No action. Behaviour looks like the typical normal bidder.",
    }[tier]


# ----------------------------------------------------------------------------
# Page header + data
# ----------------------------------------------------------------------------
st.title("🔎 Shill Bidding Review Assistant")
st.caption("A prototype that turns bidder behaviour data into a ranked review queue for a "
           "Trust & Safety analyst, using the team's Random Forest model.")

raw_df, load_message = load_training_data()
if raw_df is None:
    st.warning(f"Place **{DATA_FILE}** in the same folder as `app.py`, or upload it in the sidebar, "
               "to train the model and start the demo.")
    st.stop()

missing = [c for c in FEATURES + ["Auction_ID", "Class"] if c not in raw_df.columns]
if missing:
    st.error(f"The training file is missing expected columns: {missing}")
    st.stop()

art = build_artifacts(raw_df)
forest = art["forest"]
normal_ref = art["normal_median"]
suspicious_ref = art["suspicious_median"]

with st.sidebar:
    st.success(load_message)
    st.header("How this app works")
    st.markdown(
        "1. **Data** – bidder-in-auction behaviour records\n"
        "2. **Model** – Random Forest assigns a 0–1 risk score\n"
        "3. **Interpretation** – tier + the behaviours driving the score\n"
        "4. **Action** – analyst escalates, monitors or clears\n"
        "5. **Outcome** – results and a recommended business action"
    )
    st.divider()
    st.markdown(
        f"**Decision rules (from validation)**\n\n"
        f"- Score ≥ **{ESCALATE_THRESHOLD}** → Escalate\n"
        f"- Score ≥ **{MONITOR_THRESHOLD}** → Monitor\n"
        f"- Otherwise → Clear"
    )
    st.caption("Labels in the dataset are research labels, not confirmed fraud findings.")

tab_queue, tab_whatif, tab_evidence = st.tabs(
    ["📋 Review queue workflow", "🎛️ Single-bidder what-if", "📊 Model evidence & limits"]
)

# ============================================================================
# TAB 1  -  Review queue workflow
# ============================================================================
with tab_queue:
    st.subheader("Step 1 · Choose the business data to review")
    source = st.radio(
        "Bidding records to score",
        ["Held-out test auctions (demo — includes known research labels)",
         "Upload new bidding records (no labels needed)"],
        horizontal=True,
    )

    queue_df = None
    has_labels = False
    if source.startswith("Held-out"):
        queue_df = art["test_df"].reset_index(drop=True)
        has_labels = True
        st.caption("These auctions were never used to train or tune the model, so they stand in for "
                   "'new' activity. Labels are hidden from the model and revealed only in Step 5.")
    else:
        up = st.file_uploader("CSV with the nine behaviour columns "
                              "(Auction_ID and Bidder_ID optional, Class optional)", type="csv",
                              key="queue_upload")
        if up is not None:
            new_df = pd.read_csv(up)
            lacking = [c for c in FEATURES if c not in new_df.columns]
            if lacking:
                st.error(f"Missing required columns: {lacking}")
            else:
                queue_df = new_df.copy()
                if "Auction_ID" not in queue_df:
                    queue_df["Auction_ID"] = np.arange(len(queue_df))
                if "Bidder_ID" not in queue_df:
                    queue_df["Bidder_ID"] = [f"row{i}" for i in range(len(queue_df))]
                has_labels = "Class" in queue_df.columns
                queue_df = queue_df.reset_index(drop=True)

    if queue_df is None:
        st.info("Upload a CSV to continue, or switch to the held-out test auctions.")
    else:
        # ---- Step 2: model output ------------------------------------------------
        scores = forest.predict_proba(queue_df[FEATURES])[:, 1]
        queue_df = queue_df.assign(Risk_Score=scores, Tier=[tier_of(s) for s in scores])

        st.subheader("Step 2 · Model output: risk scores")
        n_total = len(queue_df)
        n_escalate = int((queue_df["Tier"] == "Escalate").sum())
        n_monitor = int((queue_df["Tier"] == "Monitor").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records scored", f"{n_total:,}")
        c2.metric("Escalate (≥ 0.7)", n_escalate)
        c3.metric("Monitor (0.3–0.7)", n_monitor)
        c4.metric("Clear (< 0.3)", n_total - n_escalate - n_monitor)

        # ---- Step 3: interpretation ---------------------------------------------
        st.subheader("Step 3 · Interpretation: how much can we review?")
        st.markdown(
            "Analyst time is limited, so the model's job is to **rank** records so the riskiest "
            "are reviewed first. Set the team's review capacity for this batch."
        )
        default_capacity = max(1, math.ceil(0.10 * n_total))
        capacity = st.slider("Review capacity (records the team can inspect)", 1, min(n_total, 500),
                             min(default_capacity, 500), key=f"capacity_{source[:4]}_{n_total}")

        if capacity < n_escalate:
            st.warning(f"Capacity ({capacity}) is smaller than the number of Escalate-tier records "
                       f"({n_escalate}). The queue below is ordered by score, so the highest-risk "
                       "records are covered first — consider adding analyst hours for the rest.")
        elif capacity > n_escalate + n_monitor:
            st.info("Capacity exceeds all Escalate + Monitor records; the extra slots are going to "
                    "low-risk records and add little value.")

        order = np.argsort(-queue_df["Risk_Score"].to_numpy(), kind="stable")[:capacity]
        top = queue_df.iloc[order].copy()

        drops = driver_table(forest, top[FEATURES], normal_ref)
        top["Main driver"] = [top_driver_label(drops.loc[i]) for i in top.index]
        top.insert(0, "Rank", np.arange(1, len(top) + 1))

        # ---- Step 4: user action -------------------------------------------------
        st.subheader("Step 4 · Action: record your decisions")
        st.markdown(
            "The **Decision** column is pre-filled with the model's suggested action. "
            "Override it where your judgement differs (for example, if you recognise the seller), "
            "then add optional notes."
        )

        editor_df = pd.DataFrame({
            "Rank": top["Rank"].to_numpy(),
            "Auction ID": top["Auction_ID"].to_numpy(),
            "Bidder ID": top["Bidder_ID"].astype(str).to_numpy(),
            "Risk score": top["Risk_Score"].round(3).to_numpy(),
            "Suggested": top["Tier"].to_numpy(),
            "Main driver": top["Main driver"].to_numpy(),
            "Decision": top["Tier"].to_numpy(),
            "Notes": [""] * len(top),
        })

        sig = (source[:4], n_total, capacity)
        if st.session_state.get("queue_sig") != sig:
            st.session_state["queue_sig"] = sig
            st.session_state["submitted"] = False

        edited = st.data_editor(
            editor_df,
            hide_index=True,
            key=f"editor_{sig}",
            disabled=["Rank", "Auction ID", "Bidder ID", "Risk score", "Suggested", "Main driver"],
            column_config={
                "Risk score": st.column_config.NumberColumn(format="%.3f"),
                "Decision": st.column_config.SelectboxColumn(
                    options=["Escalate", "Monitor", "Clear"], required=True),
                "Notes": st.column_config.TextColumn(),
            },
        )

        with st.expander("Why is a record ranked high? (inspect one record)"):
            pick = st.selectbox("Record", editor_df["Rank"].tolist(),
                                format_func=lambda r: f"Rank {r} · Auction "
                                f"{editor_df.loc[editor_df['Rank'] == r, 'Auction ID'].iloc[0]} · "
                                f"score {editor_df.loc[editor_df['Rank'] == r, 'Risk score'].iloc[0]:.3f}")
            idx = top.index[pick - 1]
            drv = drops.loc[idx].sort_values()
            st.markdown("**Score reduction if each behaviour were typical of a normal bidder** "
                        "(longer bar = bigger reason for the score):")
            st.bar_chart(drv.rename(index=pretty))
            compare = pd.DataFrame({
                "This record": top.loc[idx, FEATURES].astype(float),
                "Typical normal": normal_ref,
                "Typical suspicious": suspicious_ref,
            }).round(3)
            compare.index = [pretty(i) for i in compare.index]
            st.dataframe(compare)

        if st.button("✅ Submit decisions", type="primary"):
            st.session_state["submitted"] = True

        # ---- Step 5: outcome -----------------------------------------------------
        if st.session_state.get("submitted"):
            st.subheader("Step 5 · Outcome and recommendation")
            decisions = edited["Decision"].value_counts().reindex(
                ["Escalate", "Monitor", "Clear"], fill_value=0)
            d1, d2, d3 = st.columns(3)
            d1.metric("Escalated for investigation", int(decisions["Escalate"]))
            d2.metric("Placed on watchlist", int(decisions["Monitor"]))
            d3.metric("Cleared", int(decisions["Clear"]))

            reviewed = top.reset_index(drop=True).assign(Decision=edited["Decision"].to_numpy(),
                                                          Notes=edited["Notes"].to_numpy())
            escalated = reviewed[reviewed["Decision"] == "Escalate"]

            with st.expander("Business assumptions for the value estimate (editable)", expanded=False):
                a1, a2 = st.columns(2)
                cost_review = a1.number_input("Analyst cost per record reviewed ($)", 0.0, 1000.0, 25.0, 5.0)
                value_case = a2.number_input("Loss avoided per genuine shill case caught ($)", 0.0, 10000.0, 200.0, 25.0)
                st.caption("These are illustrative inputs, not figures from the dataset. Change them to "
                           "match your marketplace.")

            # Expected confirmations
            if has_labels:
                labels = reviewed["Class"].astype(int)
                confirmed = int(((reviewed["Decision"] == "Escalate") & (labels == 1)).sum())
                false_alarms = len(escalated) - confirmed
                wrongly_cleared = int(((reviewed["Decision"] == "Clear") & (labels == 1)).sum())
                total_susp = int(queue_df["Class"].sum())
                caught_in_queue = int(labels.sum())
                outside_queue = total_susp - caught_in_queue
                random_expected = capacity * (total_susp / n_total)

                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Escalations that were labelled suspicious", f"{confirmed} / {len(escalated)}")
                o2.metric("False alarms", false_alarms)
                o3.metric("Suspicious labels in reviewed queue", f"{caught_in_queue} / {total_susp}")
                o4.metric("vs. random review of same size", f"{random_expected:.1f} expected",
                          delta=f"{caught_in_queue - random_expected:+.1f} more found")
                if wrongly_cleared:
                    st.warning(f"{wrongly_cleared} record(s) you cleared were labelled suspicious — "
                               "worth a second look.")
                if outside_queue:
                    st.caption(f"{outside_queue} suspicious-labelled record(s) fell outside the review "
                               "capacity and were not inspected.")
                net_model = confirmed * value_case - capacity * cost_review
                net_random = random_expected * value_case - capacity * cost_review
                est_text = (f"Using your decisions, net value ≈ **${net_model:,.0f}** versus "
                            f"≈ **${net_random:,.0f}** if the same effort were spent on randomly chosen records.")
                precision_note = confirmed / len(escalated) if len(escalated) else 0.0
            else:
                p_hat = art["test_metrics"]["Precision"]
                confirmed_est = len(escalated) * p_hat
                net_model = confirmed_est * value_case - capacity * cost_review
                est_text = (f"No labels were supplied, so the estimate uses the model's held-out precision "
                            f"({p_hat:.1%}): about **{confirmed_est:.0f}** of the {len(escalated)} escalations "
                            f"would be genuine, for a net value ≈ **${net_model:,.0f}**.")
                precision_note = p_hat

            # Auction-level action list
            st.markdown("**Auctions to route to seller review** (any auction with an escalated bidder):")
            if len(escalated):
                auction_actions = (escalated.groupby("Auction_ID")
                                   .agg(Escalated_bidders=("Bidder_ID", "count"),
                                        Highest_score=("Risk_Score", "max"))
                                   .sort_values(["Escalated_bidders", "Highest_score"], ascending=False)
                                   .round(3))
                st.dataframe(auction_actions)
            else:
                st.caption("No records were escalated.")

            st.markdown(est_text)

            # Recommendation
            if len(escalated):
                rec = (
                    f"**Recommended business action:** route the **{len(escalated)} escalated "
                    f"bidder-auction pairs** ({escalated['Auction_ID'].nunique()} auctions) to manual "
                    f"verification, and add the **{int(decisions['Monitor'])}** watchlist records to a "
                    "re-score after the next batch. Hold any account penalty until a human confirms the "
                    f"pattern — roughly {precision_note:.0%} of escalations are expected to be genuine, so "
                    "a small share will be false alarms. Because the model leans heavily on "
                    "**Successive Outbidding**, review the non-escalated records with high Bidding Ratio "
                    "or Winning Ratio manually, as the model is much weaker when that signal is absent."
                )
            else:
                rec = ("**Recommended business action:** nothing was escalated, so no investigation is "
                       "needed for this batch. Keep the watchlist and re-score at the next batch.")
            st.success(rec)

            out_csv = reviewed.drop(columns=[c for c in ["Class"] if c in reviewed.columns]) \
                [["Rank", "Auction_ID", "Bidder_ID", "Risk_Score", "Tier", "Main driver", "Decision", "Notes"]]
            st.download_button("⬇️ Download decision log (CSV)", out_csv.to_csv(index=False),
                               file_name="shill_review_decisions.csv", mime="text/csv")

# ============================================================================
# TAB 2  -  Single-bidder what-if
# ============================================================================
with tab_whatif:
    st.subheader("Score one bidder profile")
    st.markdown("Enter or adjust a bidder's behaviour indicators to see the model's risk score, why it "
                "scored that way, and what the analyst should do next. Start from a preset, then change "
                "sliders to see which behaviours move the score.")

    def snap_duration(v):
        return int(min(DURATION_CHOICES, key=lambda d: abs(d - v)))

    def apply_preset(ref):
        for f in FEATURES:
            if f == "Auction_Duration":
                st.session_state[f"wi_{f}"] = snap_duration(ref[f])
            else:
                st.session_state[f"wi_{f}"] = float(np.clip(round(float(ref[f]), 2), 0.0, 1.0))

    for f in FEATURES:  # initial defaults = typical normal bidder
        if f"wi_{f}" not in st.session_state:
            apply_preset(normal_ref)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Typical normal bidder", on_click=apply_preset, args=(normal_ref,))
    b2.button("Typical suspicious bidder", on_click=apply_preset, args=(suspicious_ref,))

    left, right = st.columns([1, 1])
    with left:
        for f in FEATURES:
            if f == "Auction_Duration":
                st.select_slider(pretty(f) + " (days)", options=DURATION_CHOICES,
                                 key=f"wi_{f}", help=FEATURE_HELP[f])
            else:
                st.slider(pretty(f), 0.0, 1.0, step=0.01, key=f"wi_{f}", help=FEATURE_HELP[f])

    profile = pd.DataFrame([{f: st.session_state[f"wi_{f}"] for f in FEATURES}])
    score = float(forest.predict_proba(profile)[:, 1][0])
    tier = tier_of(score)
    w_drops = driver_table(forest, profile, normal_ref).iloc[0].sort_values()

    with right:
        st.metric("Risk score", f"{score:.3f}")
        if tier == "Escalate":
            st.error(f"Tier: **Escalate** (score ≥ {ESCALATE_THRESHOLD})")
        elif tier == "Monitor":
            st.warning(f"Tier: **Monitor** ({MONITOR_THRESHOLD} ≤ score < {ESCALATE_THRESHOLD})")
        else:
            st.success(f"Tier: **Clear** (score < {MONITOR_THRESHOLD})")

        st.markdown("**What the analyst does next**")
        st.write(next_step_text(tier))

        st.markdown("**Interpretation: what drives this score**")
        positive = w_drops[w_drops > 0.01].sort_values(ascending=False)
        if len(positive):
            reasons = ", ".join(f"{pretty(k)} (−{v:.2f})" for k, v in positive.head(3).items())
            st.write(f"The score would fall most if these were typical of normal bidders: {reasons}.")
        else:
            st.write("No single behaviour is pushing the score up noticeably.")
        st.bar_chart(w_drops.rename(index=pretty))
        st.caption("The score is a ranking signal from the Random Forest, not a calibrated probability "
                   "of fraud.")

# ============================================================================
# TAB 3  -  Model evidence & limits
# ============================================================================
with tab_evidence:
    st.subheader("Where the model and thresholds come from")
    st.markdown(
        "The app uses the notebook's selected model: a **Random Forest** (100 trees, settings from "
        "auction-grouped cross-validation) trained on the training auctions only. The 0.7 cut-off was the "
        "best validation F1; 0.3 is kept as a high-recall 'Monitor' line."
    )

    st.markdown("**Validation results at each threshold (Random Forest, 9 features)**")
    st.dataframe(art["validation_thresholds"].drop(columns=["Records"]).round(3))

    tm, tmm = art["test_metrics"], art["test_metrics_monitor"]
    st.markdown("**Final held-out test results**")
    ev = pd.DataFrame({
        f"Escalate rule (≥ {ESCALATE_THRESHOLD})": tm,
        f"Monitor rule (≥ {MONITOR_THRESHOLD})": tmm,
    }).T[["Precision", "Recall", "F1", "ROC AUC", "Flagged records", "Records"]]
    st.dataframe(ev.round(3))

    st.markdown("**Which features matter most (Random Forest importance)**")
    st.bar_chart(art["importance"].rename(index=pretty))
    st.caption("Importance has no direction and is shared among correlated features. "
               "It does not prove that a behaviour causes shill bidding.")

    st.markdown("**Reliance on one strong signal**")
    e8 = art["eight_feature_test"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Test F1 with all 9 features", f"{tm['F1']:.3f}")
    s2.metric("Test F1 without Successive Outbidding", f"{e8['F1']:.3f}",
              delta=f"{e8['F1'] - tm['F1']:+.3f}")
    s3.metric("Test precision without it (cut-off 0.3)", f"{e8['Precision']:.3f}")
    st.markdown(
        "Performance drops sharply without **Successive Outbidding**, so strong scores may partly reflect "
        "how the research labels were produced. Treat results as agreement with those labels, not "
        "confirmed fraud."
    )

    st.markdown("**Auction-duration experiment (not used in the workflow)**")
    st.dataframe(art["duration_results"].round(3))
    st.markdown(
        "A regression on per-auction averages did **not** beat the simple training-average baseline, so "
        "auction duration is shown only as an input feature to the risk model and is **not** forecast "
        "in this app."
    )

    st.markdown("**Limitations to keep in mind**")
    st.markdown(
        "- Labels are research labels; masked bidder IDs are not verified individuals.\n"
        "- Indicators summarise completed auctions, so scoring is retrospective, not real-time.\n"
        "- False alarms are possible; always have a human verify before any account action.\n"
        "- The model is weaker when Successive Outbidding is absent."
    )
