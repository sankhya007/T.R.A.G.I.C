from _helpers import run_script, ROOT

def test_ca_smoke():
    result = run_script("CA_evacuation.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCORE" in result.stdout
    assert (ROOT / "output" / "ca_paths.png").exists()
    assert (ROOT / "output" / "ca_report.txt").exists()