from _helpers import run_script, ROOT

def test_sfm_smoke():
    result = run_script("SFM_evacuation.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCORE" in result.stdout
    assert (ROOT / "output" / "sfm_agent_paths.png").exists()
    assert (ROOT / "output" / "SFM_output_report.txt").exists()