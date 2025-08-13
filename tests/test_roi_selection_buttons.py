import pathlib

def test_start_stop_and_no_tool_buttons():
    html = pathlib.Path('templates/roi_selection.html').read_text()
    assert 'id="startBtn"' in html
    assert 'id="stopBtn"' in html
    assert 'id="pickToolBtn"' not in html
    assert 'id="rectToolBtn"' not in html
