"""Streamlit app: credit default model, approval policy, expected loss, stress testing and monitoring.

Run with:  streamlit run app.py
"""
import altair as alt
import pandas as pd
import streamlit as st

import numpy as np

from creditrisk import (explain, fairness, features, loss, macro, metrics, model, monitoring, pipeline, quality,
                        scorecard, stress, threshold, warehouse)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"

st.set_page_config(page_title="Credit risk model lifecycle", layout="wide")


@st.cache_resource(show_spinner="Training four models on 21,000 accounts...")
def train_all():
    """Fit every model once and score the 9,000 held-out accounts."""
    return pipeline.run()


@st.cache_data
def portfolio_queries():
    con = warehouse.connect()
    return {name: warehouse.query(con, name) for name in ["03_portfolio_by_delinquency.sql", "04_portfolio_by_utilization.sql"]}


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching Federal Reserve data...")
def fed_data():
    """Pull the Fed series at most once a day; fall back to the saved snapshot if the pull fails."""
    return macro.load(live=True)


res = train_all()
fitted, scores, y_test, n_train = res.fitted, res.scores, res.y_test, len(res.X_train)
X_test = res.X_test

# ----------------------------------------------------------------- sidebar
st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Scoring model", list(model.MODELS))

st.sidebar.header("Loan economics (assumptions)")
loan_amount = st.sidebar.number_input("Amount lent per account", min_value=100, value=10_000, step=1_000)
margin_pct = st.sidebar.slider("Profit on a loan that repays (% of amount)", 1, 30, 10)
lgd_pct = st.sidebar.slider("Loss given default, LGD (% of exposure)", 10, 100, 60)
ccf_pct = st.sidebar.slider("Share of unused credit line drawn before default, CCF (%)", 0, 100, 50)
margin, lgd, ccf = margin_pct / 100, lgd_pct / 100, ccf_pct / 100
breakeven = threshold.breakeven_pd(margin, lgd)

st.sidebar.header("Approval policy")
cutoff_pct = st.sidebar.slider(
    "Approve when predicted default probability is below (%)",
    min_value=1, max_value=100, value=int(round(breakeven * 100)),
)
cutoff = cutoff_pct / 100

pd_scores = scores[model_name]
sweep_df = threshold.sweep(y_test, pd_scores, loan_amount, margin, lgd)
now = threshold.portfolio_outcomes(y_test, pd_scores, cutoff, loan_amount, margin, lgd)
best = threshold.best_cutoff(sweep_df)
approve_all = threshold.portfolio_outcomes(y_test, pd_scores, 1.01, loan_amount, margin, lgd)

# ------------------------------------------------------------------ header
st.title("Credit risk model lifecycle")
st.caption(
    f"Models trained on {n_train:,} credit card accounts and evaluated on {len(y_test):,} accounts they never saw. "
    f"{y_test.mean():.1%} of the held-out accounts defaulted. Data: UCI Default of Credit Card Clients (Taiwan, 2005)."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Approval rate", f"{now['approval_rate']:.1%}")
k2.metric("Default rate among approved", f"{now['bad_rate_approved']:.1%}")
k3.metric("Defaulters screened out", f"{now['defaults_avoided']:.1%}")
k4.metric(
    "Profit per applicant",
    f"{now['profit_per_applicant']:,.0f}",
    delta=f"{now['profit_per_applicant'] - approve_all['profit_per_applicant']:,.0f} vs approving everyone",
)

(tab_policy, tab_macro, tab_loss, tab_model, tab_monitor, tab_account, tab_sql, tab_notes) = st.tabs(
    ["Threshold trade-offs", "Current US conditions", "Expected loss and stress", "Model quality", "Monitoring and fairness",
     "Account explainer", "SQL", "Limits"]
)

# ------------------------------------------------------------ policy tab
with tab_policy:
    st.markdown(
        f"**Break-even default probability: {breakeven:.1%}.** A loan is expected to make money only below this level "
        f"(margin ÷ (margin + loss) = {margin_pct} ÷ {margin_pct + lgd_pct}). "
        f"The cutoff that earned the most on the held-out accounts was **{best:.0%}**."
    )

    plot_df = sweep_df.assign(cutoff_pct=sweep_df["cutoff"] * 100)
    rules = pd.DataFrame(
        {"cutoff_pct": [cutoff * 100, breakeven * 100], "label": ["Your cutoff", "Break-even PD"]}
    )
    rule_layer = (
        alt.Chart(rules)
        .mark_rule(strokeWidth=1.5)
        .encode(
            x="cutoff_pct:Q",
            strokeDash=alt.StrokeDash(
                "label:N",
                scale=alt.Scale(domain=["Your cutoff", "Break-even PD"], range=[[1, 0], [4, 4]]),
                legend=alt.Legend(title=None, orient="top", symbolStrokeColor=GREY),
            ),
            color=alt.value(GREY),
            tooltip=["label:N", alt.Tooltip("cutoff_pct:Q", title="Cutoff (%)", format=".1f")],
        )
    )
    x_axis = alt.X("cutoff_pct:Q", title="Approval cutoff: approve when PD is below this (%)")

    left, right = st.columns(2)
    with left:
        st.subheader("Profit per applicant at each cutoff")
        profit_line = (
            alt.Chart(plot_df)
            .mark_line(color=BLUE, strokeWidth=2)
            .encode(
                x=x_axis,
                y=alt.Y("profit_per_applicant:Q", title="Profit per applicant"),
                tooltip=[
                    alt.Tooltip("cutoff_pct:Q", title="Cutoff (%)"),
                    alt.Tooltip("profit_per_applicant:Q", title="Profit per applicant", format=",.0f"),
                    alt.Tooltip("approval_rate:Q", title="Approval rate", format=".1%"),
                ],
            )
        )
        st.altair_chart((profit_line + rule_layer).interactive(bind_y=False), width="stretch")
        st.caption("Too strict and good customers are turned away; too loose and losses on defaulters outweigh the margin.")

    with right:
        st.subheader("Approval rate against default rate")
        long = plot_df.melt(
            id_vars="cutoff_pct", value_vars=["approval_rate", "bad_rate_approved"], var_name="measure", value_name="rate"
        )
        long["measure"] = long["measure"].map(
            {"approval_rate": "Approval rate", "bad_rate_approved": "Default rate, approved"}
        )
        rate_lines = (
            alt.Chart(long)
            .mark_line(strokeWidth=2)
            .encode(
                x=x_axis,
                y=alt.Y("rate:Q", title="Share of accounts", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "measure:N",
                    scale=alt.Scale(
                        domain=["Approval rate", "Default rate, approved"], range=[BLUE, ORANGE]
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("cutoff_pct:Q", title="Cutoff (%)"),
                    "measure:N",
                    alt.Tooltip("rate:Q", title="Rate", format=".1%"),
                ],
            )
        )
        plain_rules = (
            alt.Chart(rules)
            .mark_rule(strokeWidth=1.5, color=GREY)
            .encode(
                x="cutoff_pct:Q",
                strokeDash=alt.StrokeDash("label:N", scale=alt.Scale(range=[[4, 4], [1, 0]]), legend=None),
                tooltip=["label:N", alt.Tooltip("cutoff_pct:Q", title="Cutoff (%)", format=".1f")],
            )
        )
        st.altair_chart(rate_lines + plain_rules, width="stretch")
        st.caption("Vertical lines: solid is your cutoff, dashed is break-even. Approving more accounts raises the default rate in the approved book.")

    st.subheader(f"What happens to the {len(y_test):,} held-out accounts at a {cutoff:.0%} cutoff")
    c = metrics.confusion_counts(y_test, pd_scores, cutoff)
    st.dataframe(
        pd.DataFrame(
            {
                "Repaid": [c["tn"], c["fp"]],
                "Defaulted": [c["fn"], c["tp"]],
            },
            index=["Approved", "Declined"],
        ).style.format("{:,}"),
        width="content",
    )
    st.caption(
        f"Approved and defaulted ({c['fn']:,}) are the credit losses. "
        f"Declined but would have repaid ({c['fp']:,}) is the business turned away."
    )

# ------------------------------------------------------------- model tab
with tab_model:
    st.subheader("How well each model ranks risk (held-out accounts)")
    model_table = pd.DataFrame(
        [
            {
                "Model": name,
                "AUC": metrics.auc(y_test, s),
                "KS": metrics.ks_statistic(y_test, s),
                "Best profit per applicant": threshold.sweep(y_test, s, loan_amount, margin, lgd)[
                    "profit_per_applicant"
                ].max(),
            }
            for name, s in scores.items()
        ]
    )
    st.dataframe(
        model_table.style.format({"AUC": "{:.3f}", "KS": "{:.3f}", "Best profit per applicant": "{:,.0f}"}),
        hide_index=True,
    )
    st.caption(
        "AUC: chance that a random defaulter is scored riskier than a random good payer (0.5 = coin flip). "
        "KS: largest gap between the share of defaulters and the share of good payers declined at any cutoff."
    )

    champ_best = model_table.loc[model_table["Model"] == model.CHAMPION, "Best profit per applicant"].iloc[0]
    chall_best = model_table.loc[model_table["Model"] == model.CHALLENGER, "Best profit per applicant"].iloc[0]
    st.markdown(
        f"**Price of interpretability:** the gradient-boosting challenger earns {chall_best - champ_best:,.0f} more per "
        f"applicant at its best cutoff than the logistic regression ({chall_best / champ_best - 1:+.0%}). That is what the "
        "lender gives up to keep a model whose every decision can be explained term by term."
    )

    left, right = st.columns(2)
    with left:
        st.subheader(f"ROC curve: {model_name}")
        fpr, tpr = metrics.roc_points(y_test, pd_scores)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        if len(roc_df) > 600:  # thin the curve for plotting only
            roc_df = roc_df.iloc[:: len(roc_df) // 600].reset_index(drop=True)
        roc = (
            alt.Chart(roc_df)
            .mark_line(color=BLUE, strokeWidth=2)
            .encode(
                x=alt.X("fpr:Q", title="Good payers wrongly declined", axis=alt.Axis(format="%")),
                y=alt.Y("tpr:Q", title="Defaulters correctly declined", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("fpr:Q", format=".1%"), alt.Tooltip("tpr:Q", format=".1%")],
            )
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]}))
            .mark_line(color=GREY, strokeDash=[4, 4], strokeWidth=1)
            .encode(x="fpr:Q", y="tpr:Q")
        )
        st.altair_chart(diagonal + roc, width="stretch")
        st.caption("Dashed line: a model with no skill.")

    with right:
        st.subheader("Calibration: predicted against observed")
        cal = metrics.calibration_table(y_test, pd_scores)
        points = (
            alt.Chart(cal)
            .mark_circle(color=BLUE, size=90)
            .encode(
                x=alt.X("predicted_pd:Q", title="Average predicted PD in band", axis=alt.Axis(format="%")),
                y=alt.Y("observed_default_rate:Q", title="Observed default rate", axis=alt.Axis(format="%")),
                tooltip=[
                    "band:O",
                    alt.Tooltip("accounts:Q", format=","),
                    alt.Tooltip("predicted_pd:Q", format=".1%"),
                    alt.Tooltip("observed_default_rate:Q", format=".1%"),
                ],
            )
        )
        top = float(max(cal["predicted_pd"].max(), cal["observed_default_rate"].max()))
        perfect = (
            alt.Chart(pd.DataFrame({"predicted_pd": [0, top], "observed_default_rate": [0, top]}))
            .mark_line(color=GREY, strokeDash=[4, 4], strokeWidth=1)
            .encode(x="predicted_pd:Q", y="observed_default_rate:Q")
        )
        st.altair_chart(perfect + points, width="stretch")
        st.caption("Ten equal-sized score bands. Points on the dashed line mean the PD can be trusted as a probability.")

    if model_name.startswith("Logistic"):
        st.subheader("What drives the score")
        coefs = model.logit_coefficients(fitted[model_name])
        st.dataframe(coefs.style.format({"coefficient": "{:+.3f}"}), hide_index=True)
        st.caption(
            "Change in log-odds of default for a one-standard-deviation increase in each input. "
            "Positive raises predicted risk."
        )

# ------------------------------------------------------------- macro tab
with tab_macro:
    fed_long, fed_source = fed_data()
    panel = macro.quarterly_panel(fed_long)
    now_vals = macro.latest(fed_long)
    fit = macro.fit_chargeoff_model(panel)

    def quarter(ts):
        return f"{ts.year} Q{ts.quarter}"

    if fed_source == "live":
        st.success("Federal Reserve data pulled live from FRED (refreshed at most once a day).")
    else:
        st.warning("Live pull from FRED failed, so this tab is using the snapshot saved with the project "
                   f"(latest quarter {quarter(now_vals['CORCCACBS']['date'])}).")

    year_ago = panel.iloc[-5] if len(panel) > 4 else panel.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"US unemployment ({now_vals['UNRATE']['date']:%b %Y})", f"{now_vals['UNRATE']['value']:.1f}%")
    m2.metric(f"Card charge-off rate ({quarter(now_vals['CORCCACBS']['date'])})", f"{now_vals['CORCCACBS']['value']:.2f}%",
              delta=f"{now_vals['CORCCACBS']['value'] - year_ago['chargeoff']:+.2f} pts vs a year earlier", delta_color="inverse")
    m3.metric(f"Card delinquency rate ({quarter(now_vals['DRCCLACBS']['date'])})", f"{now_vals['DRCCLACBS']['value']:.2f}%",
              delta=f"{now_vals['DRCCLACBS']['value'] - year_ago['delinquency']:+.2f} pts vs a year earlier", delta_color="inverse")
    st.caption("Source: Board of Governors of the Federal Reserve System and Bureau of Labor Statistics, via FRED "
               "(series UNRATE, CORCCACBS, DRCCLACBS). Rates cover credit card loans at all US commercial banks.")

    left, right = st.columns(2)
    with left:
        st.subheader("US card credit quality since 1991")
        hist = panel.reset_index().rename(columns={"index": "date"}).melt(
            id_vars="date", value_vars=["chargeoff", "delinquency"], var_name="series", value_name="rate")
        hist["series"] = hist["series"].map({"chargeoff": "Charge-off rate", "delinquency": "Delinquency rate"})
        st.altair_chart(
            alt.Chart(hist).mark_line(strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("rate:Q", title="Percent of card balances"),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Charge-off rate", "Delinquency rate"], range=[BLUE, ORANGE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("date:T", format="%b %Y"), "series:N", alt.Tooltip("rate:Q", format=".2f")],
            ),
            width="stretch",
        )
    with right:
        st.subheader("Unemployment against charge-offs")
        changes = macro.yoy_changes(panel).reset_index().rename(columns={"index": "date"})
        xs = np.array([changes["d_unemployment"].min(), changes["d_unemployment"].max()])
        line = pd.DataFrame({"d_unemployment": xs, "d_chargeoff": fit["intercept"] + fit["slope"] * xs})
        dots = alt.Chart(changes).mark_circle(color=BLUE, size=45, opacity=0.6).encode(
            x=alt.X("d_unemployment:Q", title="One-year change in unemployment (points)"),
            y=alt.Y("d_chargeoff:Q", title="One-year change in charge-off rate (points)"),
            tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("d_unemployment:Q", format="+.2f"),
                     alt.Tooltip("d_chargeoff:Q", format="+.2f")],
        )
        fit_line = alt.Chart(line).mark_line(color=ORANGE, strokeWidth=2).encode(x="d_unemployment:Q", y="d_chargeoff:Q")
        st.altair_chart(dots + fit_line, width="stretch")
        with_covid = macro.fit_chargeoff_model(panel, exclude=None)
        st.caption(
            f"Each dot is one quarter. Fitted line: a one-point rise in unemployment over a year goes with a "
            f"{fit['slope']:.2f}-point rise in the charge-off rate (R² {fit['r2']:.2f}, {fit['n']} quarters). "
            f"2020 to 2022 is excluded because stimulus and forbearance reversed the usual link; "
            f"including those years the slope drops to {with_covid['slope']:.2f} (R² {with_covid['r2']:.2f})."
        )

    st.subheader("The sample portfolio under today's conditions and the Fed's stress scenario")
    macro_ead = loss.exposure_at_default(X_test["BILL_AMT1"], X_test["LIMIT_BAL"], ccf)
    macro_df = macro.macro_stress_table(pd_scores, macro_ead, lgd, fed_long)
    st.dataframe(
        macro_df.rename(columns={
            "scenario": "Scenario", "unemployment_pct": "Unemployment (%)", "chargeoff_rate_pct": "Card charge-off rate (%)",
            "portfolio_pd": "Portfolio PD", "expected_loss": "Expected loss", "expected_loss_vs_current": "Change vs current"}
        ).style.format({"Unemployment (%)": "{:.1f}", "Card charge-off rate (%)": "{:.2f}", "Portfolio PD": "{:.1%}",
                        "Expected loss": "{:,.0f}", "Change vs current": "{:+.0%}"}),
        hide_index=True,
    )
    fed_row, peak_row = macro_df.iloc[1], macro_df.iloc[2]
    st.markdown(
        f"The Federal Reserve's 2026 severely adverse scenario takes unemployment to "
        f"{macro.FED_SEVERELY_ADVERSE_UNEMPLOYMENT:.0f}%. The regression turns that into a card charge-off rate of "
        f"**{fed_row['chargeoff_rate_pct']:.1f}%**, close to the {peak_row['chargeoff_rate_pct']:.1f}% the industry actually "
        f"reached in {peak_row['scenario'].split('(')[1].split(',')[0]}. On this portfolio, expected loss rises "
        f"**{fed_row['expected_loss_vs_current']:+.0%}**."
    )
    st.caption(
        "How it works: each account keeps its rank from the model, and every PD is shifted by the same amount so the "
        "portfolio average equals charge-off rate ÷ LGD. The accounts are still the 2005 Taiwan sample; only the level "
        "of risk is tied to US data. Charge-off rate ≈ PD × LGD is an approximation."
    )

# -------------------------------------------------------------- loss tab
with tab_loss:
    ead = loss.exposure_at_default(X_test["BILL_AMT1"], X_test["LIMIT_BAL"], ccf)
    summ = loss.portfolio_summary(pd_scores, ead, lgd)
    realised = loss.realised_loss(y_test, ead, lgd)
    st.markdown(
        "**Expected loss = PD × LGD × EAD.** Exposure at default (EAD) is each account's balance plus the share of "
        "its unused credit line assumed to be drawn before default. Amounts are in New Taiwan dollars."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposure at default", f"{summ['exposure_at_default'] / 1e6:,.0f}M")
    c2.metric("Expected loss", f"{summ['expected_loss'] / 1e6:,.1f}M")
    c3.metric("Expected loss rate", f"{summ['expected_loss_rate']:.1%}")
    c4.metric("Loss on accounts that did default", f"{realised / 1e6:,.1f}M")
    st.caption(
        f"Outcomes check: expected loss came out {summ['expected_loss'] / realised - 1:+.1%} against the loss on the "
        "accounts that actually defaulted, using the same LGD and EAD assumptions for both."
    )

    st.subheader("Where the expected loss sits, by risk grade")
    grades = scorecard.risk_grade(pd_scores)
    by_grade = loss.loss_by_grade(grades, pd_scores, ead, lgd)
    st.dataframe(
        by_grade.rename(columns=lambda c: c.replace("_", " ").capitalize()).style.format(
            {"Accounts": "{:,}", "Average pd": "{:.1%}", "Exposure at default": "{:,.0f}",
             "Expected loss": "{:,.0f}", "Share of expected loss": "{:.1%}"}
        ),
        hide_index=True,
    )
    st.caption(
        f"Grades: A below 5% PD, B 5-10%, C 10-20%, D 20-40%, E 40% and above. "
        f"The riskiest 10% of accounts carry {summ['el_share_top_decile']:.0%} of expected loss."
    )

    st.subheader("Stress test: what if customer behaviour deteriorates?")
    stress_df = pipeline.stress_table(fitted[model_name], X_test, lgd, ccf, cutoff)
    bars = (
        alt.Chart(stress_df.assign(el_m=stress_df["expected_loss"] / 1e6))
        .mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=70)
        .encode(
            x=alt.X("scenario:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("el_m:Q", title="Expected loss (millions)"),
            tooltip=[
                "scenario:N",
                alt.Tooltip("el_m:Q", title="Expected loss (M)", format=",.1f"),
                alt.Tooltip("expected_loss_vs_baseline:Q", title="Change vs baseline", format="+.1%"),
            ],
        )
    )
    left, right = st.columns([2, 3])
    with left:
        st.altair_chart(bars, width="stretch")
    with right:
        st.dataframe(
            stress_df.rename(columns=lambda c: c.replace("_", " ").capitalize()).style.format(
                {"Average pd": "{:.1%}", "Exposure at default": "{:,.0f}", "Expected loss": "{:,.0f}",
                 "Expected loss rate": "{:.1%}", "Approval rate at cutoff": "{:.1%}",
                 "Share grade d or e": "{:.1%}", "Expected loss vs baseline": "{:+.1%}"}
            ),
            hide_index=True,
        )
        for s in stress.SCENARIOS[1:]:
            st.caption(f"**{s.name}:** {s.description}")
    st.caption(
        "These are sensitivity scenarios on the model's inputs. The data has no economic variables, so they are not "
        "forecasts tied to unemployment or interest rates."
    )

# ----------------------------------------------------------- monitor tab
with tab_monitor:
    st.subheader("Population stability: has the scored population drifted?")
    dev_scores = res.train_scores[model_name]
    rows = [{"Comparison": "Development (train) vs held-out accounts", "Score PSI": monitoring.psi(dev_scores, pd_scores)}]
    shocked = {}
    for s in stress.SCENARIOS[1:]:
        shocked[s.name] = stress.apply(X_test, s)
        rows.append({"Comparison": f"Development vs {s.name.lower()} scenario",
                     "Score PSI": monitoring.psi(dev_scores, model.predict_pd(fitted[model_name], shocked[s.name]))})
    psi_df = pd.DataFrame(rows)
    psi_df["Status"] = psi_df["Score PSI"].map(monitoring.band)
    st.dataframe(psi_df.style.format({"Score PSI": "{:.3f}"}), hide_index=True)
    st.caption("PSI below 0.10 is conventionally treated as stable, 0.10 to 0.25 as worth investigating, above 0.25 as a material shift.")

    pick = st.selectbox("Input drift under scenario", list(shocked))
    st.dataframe(
        monitoring.feature_psi(features.engineer(res.X_train), features.engineer(shocked[pick]))
        .rename(columns={"variable": "Model input", "psi": "PSI", "status": "Status"})
        .style.format({"PSI": "{:.3f}"}),
        hide_index=True,
    )
    st.caption(
        "Input-level PSI can flag a shift that the overall score PSI understates, "
        "which is why monitoring tracks both."
    )

    st.subheader(f"Outcome testing at a {cutoff:.0%} cutoff")
    st.markdown(
        "Demographics are not model inputs. They are used here only to test whether approval rates differ by group. "
        "A ratio below 0.80 against the highest-approval group is flagged for review; it is a screening convention, "
        "not a legal conclusion."
    )
    approved = pd_scores < cutoff
    cols = st.columns(2)
    for i, attribute in enumerate(res.demographics.columns):
        with cols[i % 2]:
            st.markdown(f"**{attribute}**")
            t = fairness.adverse_impact(res.demographics[attribute], approved, y_test)
            st.dataframe(
                t.rename(columns=lambda c: c.replace("_", " ").capitalize()).style.format(
                    {"Accounts": "{:,}", "Approval rate": "{:.1%}", "Observed default rate": "{:.1%}",
                     "Adverse impact ratio": "{:.2f}"}
                ),
                hide_index=True,
            )

    st.subheader("Data-quality controls")
    checks = quality.run_checks(res.raw)
    st.dataframe(checks.assign(passed=checks["passed"].map({True: "Pass", False: "FAIL"}))
                 .rename(columns={"check": "Check", "passed": "Result", "detail": "Found"}), hide_index=True)

# ----------------------------------------------------------- account tab
with tab_account:
    st.subheader("Why did this account get its score?")
    champion = fitted[model.CHAMPION]
    champ_pd = scores[model.CHAMPION]
    ids = res.raw.loc[X_test.index, "ID"].to_numpy()
    only_declined = st.checkbox("Show declined accounts only", value=True)
    options = ids[champ_pd >= cutoff] if only_declined else ids
    if len(options) == 0:
        st.info("No accounts are declined at this cutoff.")
    else:
        account_id = st.selectbox("Account ID (held-out accounts)", options[:500])
        pos = int(np.where(ids == account_id)[0][0])
        row = X_test.iloc[[pos]]
        p = float(champ_pd[pos])
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Predicted PD", f"{p:.1%}")
        a2.metric("Credit score", f"{scorecard.pd_to_score(p):.0f}")
        a3.metric("Risk grade", scorecard.risk_grade([p])[0])
        a4.metric("Decision", "Approve" if p < cutoff else "Decline")

        contrib = explain.contributions(champion, row).iloc[0]
        cdf = contrib.rename("contribution").reset_index().rename(columns={"index": "input"})
        cdf["effect"] = np.where(cdf["contribution"] >= 0, "Raises risk", "Lowers risk")
        bars = (
            alt.Chart(cdf)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("contribution:Q", title="Contribution to log-odds of default (vs the average account)"),
                y=alt.Y("input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color("effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[ORANGE, BLUE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["input:N", alt.Tooltip("contribution:Q", format="+.3f"), "effect:N"],
            )
        )
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(bars, width="stretch")
        with right:
            st.markdown("**Principal reasons** (factors that raised this account's risk, strongest first)")
            reasons = explain.reason_codes(champion, row)[0]
            for n, text in enumerate(reasons, 1):
                st.markdown(f"{n}. {text}")
            if not reasons:
                st.markdown("No input pushed this account above average risk.")
            rebuilt = explain.pd_from_contributions(champion, contrib)
            st.caption(f"Check: intercept plus the bars rebuilds a PD of {rebuilt:.1%}, matching the model.")
        st.caption("Explanations always use the logistic regression, because its score is an exact sum of these terms.")

# --------------------------------------------------------------- sql tab
with tab_sql:
    st.markdown(
        "These tables come from SQL run against all 30,000 accounts in an in-process DuckDB database. "
        "The model inputs are also defined as a SQL view, and a test confirms it matches the Python version row for row."
    )
    for name, title in [("03_portfolio_by_delinquency.sql", "Portfolio by current delinquency"),
                        ("04_portfolio_by_utilization.sql", "Portfolio by credit line utilization")]:
        st.subheader(title)
        st.dataframe(portfolio_queries()[name].style.format(thousands=",", precision=1), hide_index=True)
        with st.expander("Show the SQL"):
            st.code(warehouse.read_sql(name), language="sql")
    with st.expander("Show the feature view SQL"):
        st.code(warehouse.read_sql("02_account_features.sql"), language="sql")

# ------------------------------------------------------------- notes tab
with tab_notes:
    st.markdown(
        """
- **Existing accounts, not new applicants.** The data describes cardholders who already had six months of history,
  so this is closer to a behavioural score (line increases, renewals) than a new-application score.
- **One market, one period.** Taiwan, April to September 2005, during a local card-debt crisis: 22% of accounts
  defaulted, far above a normal card portfolio. Cutoffs here would not transfer to another book.
- **Profit inputs are assumptions.** On the threshold tab every account is treated as the same size, with one margin
  and one loss rate. LGD, CCF and the risk-grade bands are also assumptions, not estimates from data.
- **Stress scenarios are hypothetical.** The scenarios on the stress tab shock the model's inputs. The Fed-anchored
  scenario on the conditions tab rests on a one-variable regression that explains about half the variation in
  charge-offs and leaves out 2020 to 2022.
- **US re-basing changes the level, not the borrowers.** The accounts remain the Taiwan sample; "charge-off rate ≈
  PD × LGD" is an approximation, and the Fed data is published quarterly, so "live" means the latest release.
- **Reason codes need review.** The model scores "paid in full this month" as riskier than carrying a balance, a
  counter-intuitive sign that a validator would challenge before these reasons went to a customer.
- **The "best" cutoff is picked on the same held-out accounts it is reported on,** so its profit is somewhat flattering.
- **No demographic inputs.** Sex, age, marital status and education are excluded from the model and used only for
  outcome testing. Groups under 500 accounts are not assessed.
- **Amounts are in New Taiwan dollars** in the source data; the loan amount in the sidebar is unit-free.
"""
    )
