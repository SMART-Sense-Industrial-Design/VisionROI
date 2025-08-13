import pathlib

def test_no_start_stop_buttons():
    html = pathlib.Path('templates/roi_selection.html').read_text()
    assert 'id="startBtn"' not in html
    assert 'id="stopBtn"' not in html
