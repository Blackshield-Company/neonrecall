import json

from neonrecall.cli import main
from tests.conftest import CORPUS, QA


def test_cli_ingest_search_ask(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["ingest", str(CORPUS), "--base", str(CORPUS), "--db", db, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 8

    assert main(["search", "degaussing wand", "-k", "2", "--db", db]) == 0
    assert "crt-maintenance.md" in capsys.readouterr().out

    assert main(["ask", "How many pulses per quarter note does MIDI clock send?", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "24" in out and "midi-sync.md" in out

    assert main(["stats", "--db", db, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] > 0


def test_cli_eval_thresholds(capsys):
    assert main(["eval", "--corpus", str(CORPUS), "--qa", str(QA), "--min-recall", "0.8"]) == 0
    assert "hybrid" in capsys.readouterr().out
    assert main(["eval", "--corpus", str(CORPUS), "--qa", str(QA), "--min-recall", "1.01"]) == 1
