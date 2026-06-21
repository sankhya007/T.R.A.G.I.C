from _helpers import run_script, ROOT

def test_rvo_smoke():
    result = run_script("RVO_evacuation.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCORE" in result.stdout
    assert (ROOT / "output" / "rvo_agent_paths.png").exists()
    assert (ROOT / "output" / "RVO_output_report.txt").exists()