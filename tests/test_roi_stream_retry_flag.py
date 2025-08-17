import re


def test_start_stream_resets_flag_on_fail():
    with open("templates/roi_selection.html", encoding="utf-8") as f:
        content = f.read()

    # hasStarted must reset when name is missing
    assert re.search(r"if \(!name\) \{[^}]*hasStarted = false;", content, re.DOTALL)

    # hasStarted must reset when starting ROI stream fails
    assert re.search(r"if \(!startRes\.ok\) \{[^}]*hasStarted = false;", content, re.DOTALL)

