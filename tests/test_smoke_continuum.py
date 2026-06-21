from _helpers import run_script, ROOT

def test_continuum_smoke():
    result = run_script("continuum_evacuation_path.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCORE" in result.stdout
    assert (ROOT / "output" / "continuum_agent_paths.png").exists()
    assert (ROOT / "output" / "continuum_report.txt").exists()