"""Write the CSV extracts a BI tool (Tableau Public, Power BI) can connect to.

Run from the project folder:  python scripts/build_exports.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from creditrisk import model, pipeline, threshold  # noqa: E402

CUTOFF, LGD, CCF, MARGIN, LOAN = 0.14, 0.60, 0.50, 0.10, 10_000


def main():
    out = ROOT / "exports"
    out.mkdir(exist_ok=True)
    res = pipeline.run()
    name = model.CHAMPION

    accounts = pipeline.scored_accounts(res, name, CUTOFF, LGD, CCF)
    accounts.to_csv(out / "accounts_scored.csv", index=False)

    sweep = threshold.sweep(res.y_test, res.scores[name], LOAN, MARGIN, LGD)
    sweep.round(4).to_csv(out / "threshold_sweep.csv", index=False)

    stress = pipeline.stress_table(res.fitted[name], res.X_test, LGD, CCF, CUTOFF)
    stress.round(4).to_csv(out / "stress_scenarios.csv", index=False)

    print(f"Wrote {len(accounts):,} scored accounts, {len(sweep)} cutoffs and {len(stress)} scenarios to {out}")


if __name__ == "__main__":
    main()
