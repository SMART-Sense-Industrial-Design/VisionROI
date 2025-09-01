import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_click_creates_page_rect_and_saves():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"canvas.addEventListener\('click',\s*\(e\) => \{([\s\S]*?drawAllRois\(\);\n\s*)\}\);", html)
    assert match, 'click handler not found'
    handler = match.group(1)

    script = textwrap.dedent("""
    let rois = [];
    let pageRois = [];
    let otherRois = [];
    let rectStart = null, rectEnd = null, drawingRect = false;
    let currentPoints = [];
    let currentMode = 'rect';
    let currentType = 'page';
    let currentSource = 'src';
    let fetchBody;
    function renderRoiList(){}
    function drawAllRois(){}
    Date.now = () => 123;
    global.fetch = (url, opts) => { fetchBody = opts.body; return Promise.resolve({}); };
    const canvas = { width:100, height:100, getBoundingClientRect: () => ({left:0, top:0, width:100, height:100}) };
    function handler(e) {
{handler}
    }
    handler({clientX:10, clientY:20});
    handler({clientX:50, clientY:60});
    console.log(JSON.stringify({pageRois, fetchBody}));
    """).replace('{handler}', handler)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout.strip())
    assert len(data['pageRois']) == 1
    pts = data['pageRois'][0]['points']
    assert pts[0] == {'x': 10, 'y': 20}
    assert pts[1] == {'x': 50, 'y': 20}
    assert pts[2] == {'x': 50, 'y': 60}
    assert pts[3] == {'x': 10, 'y': 60}
    assert data['pageRois'][0]['id'] == 'roi_123'
    assert data['pageRois'][0]['page'] == ''
    assert data['pageRois'][0]['image'] is None
    payload = json.loads(data['fetchBody'])
    assert payload['rois'][0]['points'] == pts
    assert payload['rois'][0]['id'] == 'roi_123'
    assert payload['rois'][0]['page'] == ''
    assert payload['rois'][0]['type'] == 'page'
