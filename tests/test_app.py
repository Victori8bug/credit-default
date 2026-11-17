"""Smoke test: the app runs start to finish and responds to the sidebar controls."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_and_reacts_to_cutoff():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].label == "Approval rate"
    before = at.metric[0].value

    cutoff = next(sl for sl in at.sidebar.slider if sl.label.startswith("Approve when"))
    cutoff.set_value(50).run()  # loosen the approval cutoff to 50%
    assert not at.exception
    assert float(at.metric[0].value.rstrip("%")) > float(before.rstrip("%"))


def test_every_model_option_renders():
    at = AppTest.from_file(APP, default_timeout=120).run()
    for option in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value(option).run()
        assert not at.exception
