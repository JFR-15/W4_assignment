"""Semester at the study office: who should the three advisers talk to at the end of week 6?

The model sits next to the code, in model/: booster.json (XGBoost) + preprocess.json (filling gaps, scaling and
one-hot as plain numbers), written by portable.export() in step 9 of the notebook. It was trained on the 2023 and
2024 cohorts only, so its mistakes on 2025 are an honest preview of how it will do on this year's students.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from portable import Model

MODEL_DIR = Path(__file__).parent / "model"
URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
CAPACITY = 40
LABEL = {"logins_total": "{v:.0f} logins in weeks 1–6", "logins_last3": "{v:.0f} logins in the last 3 weeks",
         "logins_trend": "login trend {v:+.1f}", "submitted_share": "{v:.0%} of assignments handed in",
         "missed_last3": "{v:.0f} assignments missed lately", "quiz_mean": "quiz average {v:.0f}",
         "weeks_since_login": "{v:.0f} weeks since last login", "fees_owed": "owes fees: {v:.0f}",
         "su_scholarship": "SU scholarship: {v:.0f}", "age": "age {v:.0f}", "admission_grade": "admission grade {v:.1f}",
         "international": "international: {v:.0f}", "first_gen": "first in family: {v:.0f}",
         "moved_from_home": "moved from home: {v:.0f}", "married": "married: {v:.0f}",
         "evening_programme": "evening programme: {v:.0f}", "programme": "programme: {v}", "gender": "gender: {v}"}

st.set_page_config(page_title="Semester at the study office", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- the model and the data, loaded once per server
@st.cache_resource
def load():
    pipe = Model(MODEL_DIR)                                  # no pickle: loads with any recent pandas/xgboost
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    val = history[history["cohort"] == 2025].copy()          # the cohort the model never saw
    val["risk"] = pipe.predict_proba(val)
    new["risk"] = pipe.predict_proba(new)
    return pipe, val, new


def reasons(pipe, rows, top=2):
    """The features that push each student's risk up most (XGBoost's own SHAP contributions)."""
    by_f = pipe.contributions(rows).reset_index(drop=True)
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        best = by_f.iloc[i].sort_values(ascending=False).head(top)
        out.append(" · ".join(f"{LABEL.get(f, f + ': {v}').format(v=row[f])} ↑" for f, c in best.items() if c > 0))
    return out


def mistakes(df, n):
    """The four boxes when the office talks to the n students with the highest risk."""
    on_list = df["risk"].rank(ascending=False) <= n
    reached = int((on_list & (df["left"] == 1)).sum())
    worried = int((on_list & (df["left"] == 0)).sum())
    missed = int((~on_list & (df["left"] == 1)).sum())
    fine = int((~on_list & (df["left"] == 0)).sum())
    return {"reached": reached, "worried": worried, "missed": missed, "fine": fine,
            "precision": reached / max(reached + worried, 1), "recall": reached / max(reached + missed, 1)}


pipe, val, new = load()

with st.sidebar:
    st.header("🎛️ The rule")
    n = st.slider("Conversations at week 6", 10, 200, CAPACITY, 5,
                  help="The office talks to this many students, starting with the highest risk. Three advisers ≈ 40.")
    st.header("💶 What things cost (DKK)")
    cost_talk = st.number_input("One conversation (adviser time)", 0, 5_000, 500, 100)
    cost_worry = st.number_input("One student worried for nothing", 0, 20_000, 2_000, 500)
    cost_leave = st.number_input("One student who leaves", 0, 200_000, 60_000, 5_000)
    helps = st.slider("Share of at-risk students a conversation keeps", 0.0, 1.0, 0.30, 0.05)
    st.caption("These are assumptions. The study office and management should set them, not the data team.")

st.title("🎓 Semester at the study office")
st.markdown(f"Who should the advisers talk to at the end of week 6? The model ranks this year's students by their "
            f"risk of leaving. **A person decides**: the list is a starting point, not a verdict.")
tab_list, tab_rule, tab_group, tab_hood = st.tabs(["📋 This week's list", "⚖️ The mistakes of the rule",
                                                    "🌍 Per group", "⚙️ How it works"])

# ---------------------------------------------------------------- 1. this week's list
with tab_list:
    ranked = new.sort_values("risk", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["talk this week"] = ranked["rank"] <= n
    c1, c2, c3 = st.columns(3)
    c1.metric("Students at week 6", len(new))
    c2.metric("Expected to leave (sum of risks)", f"{new['risk'].sum():.0f}")
    c3.metric(f"Risk of student no. {n}", f"{ranked['risk'].iloc[n - 1]:.0%}")

    show = ranked.head(n).copy()
    show["why"] = reasons(pipe, show)
    show["risk"] = (100 * show["risk"]).round()
    st.markdown(f"**The {n} students to talk to this week**, highest risk first. *Why* lists what pushes each "
                f"student's risk up most. Read it before reaching out: some students are on the list because of "
                f"who they are (programme, age), not what they do.")
    st.dataframe(show[["rank", "student_id", "risk", "why", "programme", "international", "fees_owed",
                       "logins_total", "submitted_share", "quiz_mean"]],
                 column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%"),
                                "submitted_share": st.column_config.NumberColumn("handed in", format="%.2f"),
                                "logins_total": "logins wk 1–6", "fees_owed": "owes fees"},
                 hide_index=True, width="stretch")
    with st.expander("All students this week"):
        ranked["risk"] = (100 * ranked["risk"]).round()
        st.dataframe(ranked[["rank", "talk this week", "student_id", "risk", "programme", "international"]],
                     column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%")},
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- 2. the mistakes of the rule, on 2025
with tab_rule:
    m = mistakes(val, n)
    st.markdown(f"**If the office had used this rule last year** ({len(val)} students in the 2025 cohort, "
                f"{int(val['left'].sum())} of whom left):")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Reached in time", m["reached"], help="On the list, and really about to leave.")
    c2.metric("📞 Worried for nothing", m["worried"], help="On the list, but would have stayed anyway.")
    c3.metric("🚪 Missed", m["missed"], help="Left, and nobody talked to them.")
    st.info(f"Of the **{n} students** the advisers talk to, **{m['reached']}** were really about to leave "
            f"(precision {m['precision']:.0%}). Of all **{m['reached'] + m['missed']}** students who left, the "
            f"office reached **{m['reached']}** (recall {m['recall']:.0%}). **{m['worried']}** students got a "
            f"worrying message they did not need, and **{m['missed']}** left without anyone trying.")

    value = m["reached"] * helps * cost_leave - n * cost_talk - m["worried"] * cost_worry
    st.markdown("#### 💶 Is the rule worth it?")
    st.markdown(f"With the costs in the sidebar, this rule is worth about **{value:,.0f} DKK** compared with "
                f"talking to nobody: {m['reached']} students reached × {helps:.0%} kept × {cost_leave:,} DKK, minus "
                f"{n} conversations × {cost_talk:,} DKK, minus {m['worried']} worried students × {cost_worry:,} DKK.")
    curve = pd.DataFrame([{"conversations": k,
                           "net value (DKK)": mistakes(val, k)["reached"] * helps * cost_leave - k * cost_talk
                           - mistakes(val, k)["worried"] * cost_worry} for k in range(10, 301, 5)]).set_index("conversations")
    st.line_chart(curve)
    best = int(curve["net value (DKK)"].idxmax())
    st.caption(f"With these costs, the best number of conversations would be about **{best}**. "
               f"The office can hold about {CAPACITY}.")

# ---------------------------------------------------------------- 3. per group
with tab_group:
    st.markdown(f"The same rule ({n} conversations), on the 2025 cohort, split into domestic and international students.")
    on_list = val["risk"].rank(ascending=False) <= n          # ONE list for everyone, then split by group
    rows = []
    for label, part in [("Domestic", val[val["international"] == 0]), ("International", val[val["international"] == 1])]:
        sel = on_list[part.index]
        g = {"reached": int((sel & (part["left"] == 1)).sum()), "worried": int((sel & (part["left"] == 0)).sum()),
             "missed": int((~sel & (part["left"] == 1)).sum())}
        rows.append({"group": label, "students": len(part), "really left": f"{part['left'].mean():.1%}",
                     "model's average risk": f"{part['risk'].mean():.1%}", "reached in time": g["reached"],
                     "worried for nothing": g["worried"], "missed": g["missed"],
                     "recall": f"{g['reached'] / max(g['reached'] + g['missed'], 1):.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.warning("Both groups leave about equally often, but the model gives international students a lower risk and "
               "reaches fewer of those who leave. International students log in much less in general, also the ones "
               "who are doing fine, so a login does not mean the same for everyone. Check this group by hand, and ask "
               "why before trusting logins as a warning sign.")

# ---------------------------------------------------------------- 4. how it works
with tab_hood:
    st.markdown("""
### How the list is made
- **Data:** what the office knows at the end of week 6: enrolment (age, programme, admission grade), money and life
  situation, and activity on the learning platform in weeks 1–6. Columns only known later (ECTS points, the
  deregistration form, the last login week) are left out: they would make the model look perfect and be useless now.
- **Model:** XGBoost, trained on the 2023 and 2024 cohorts and checked on 2025 (AUC about 0.78: it ranks a student
  who leaves above one who stays about 78 % of the time).
- **Rule:** talk to the students with the highest risk, as many as the advisers can hold.

### Before a university uses this
- **A person decides.** The list is a suggestion; an adviser reads each row before reaching out.
- **Students are told** that week-6 data is used to offer help, and **can object**.
- **The message is an offer, not a warning.** Different reasons need different help: money, study habits, wellbeing.
- **Check the mistakes every year, per group**, and retrain when the outcomes for a new cohort are known.
""")
