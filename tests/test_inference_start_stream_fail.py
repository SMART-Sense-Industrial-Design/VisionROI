import re


def test_inference_group_start_fail_message():
    with open("templates/partials/inference_content.html", encoding="utf-8") as f:
        content = f.read()
    assert "Failed to start inference stream" in content
    assert re.search(r"catch\s*\([^)]*\)\s*\{[^}]*running\s*=\s*false;[^}]*startButton\.disabled\s*=\s*false;", content, re.DOTALL)


def test_inference_page_start_fail_message():
    with open("templates/partials/inference_page_content.html", encoding="utf-8") as f:
        content = f.read()
    assert "Failed to start inference stream" in content
    assert re.search(r"catch\s*\([^)]*\)\s*\{[^}]*running\s*=\s*false;[^}]*startButton\.disabled\s*=\s*false;", content, re.DOTALL)
