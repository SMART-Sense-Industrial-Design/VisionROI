import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_drag_creates_rectangle_and_saves():
    html = Path('templates/roi_selection.html').read_text()
    down_match = re.search(r"frameContainer.addEventListener\('mousedown',\s*\(e\) => \{([\s\S]*?)\}\);", html)
    up_match = re.search(r"frameContainer.addEventListener\('mouseup',\s*\(e\) => \{([\s\S]*?drawAllRois\(\);\n\s*)\}\);", html)
    assert down_match and up_match, 'handlers not found'
    down_handler = down_match.group(1)
    up_handler = up_match.group(1)

    script = textwrap.dedent("""
    let rois = [];
    let modules = ['m1'];
    let rectStart = null, rectEnd = null, drawingRect = false;
    let currentPoints = [];
    let currentSource = 'src';
    let fetchBody;
    let hoverPoint = null;
    let currentTool = 'rect';
    function renderRoiList(){}
    function drawAllRois(){}
    global.prompt = (msg) => {
        if (msg === 'ROI id?') return '1';
        if (msg.startsWith('Module?')) return 'm1';
        return null;
    };
    global.fetch = (url, opts) => { fetchBody = opts.body; return Promise.resolve({}); };
    const frameContainer = { getBoundingClientRect: () => ({left:0, top:0, width:100, height:100}) };
    const canvas = { width:100, height:100 };
    function handlerDown(e) {
{down_handler}
    }
    function handlerUp(e) {
{up_handler}
    }
    handlerDown({clientX:10, clientY:20});
    handlerUp({clientX:50, clientY:60});
    console.log(JSON.stringify({rois, fetchBody}));
    """).replace('{down_handler}', down_handler).replace('{up_handler}', up_handler)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout.strip())
    assert len(data['rois']) == 1
    pts = data['rois'][0]['points']
    assert pts[0] == {'x': 10, 'y': 20}
    assert pts[1] == {'x': 50, 'y': 20}
    assert pts[2] == {'x': 50, 'y': 60}
    assert pts[3] == {'x': 10, 'y': 60}
    payload = json.loads(data['fetchBody'])
    assert payload['rois'][0]['points'] == pts
