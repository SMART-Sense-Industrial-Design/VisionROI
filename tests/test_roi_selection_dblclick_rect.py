import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_dblclick_creates_rectangle_and_saves():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"frameContainer.addEventListener\('dblclick',\s*\(e\) => \{([\s\S]*?drawAllRois\(\);\n\s*)\}\);", html)
    assert match, 'dblclick handler not found'
    handler = match.group(1)

    script = textwrap.dedent("""
    let rois = [];
    let modules = ['m1'];
    let rectStart = null, rectEnd = null, drawingRect = false;
    let currentPoints = [];
    let currentSource = 'src';
    let fetchBody;
    let hoverPoint = null;
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
    function handler(e) {
{handler}
    }
    handler({clientX:10, clientY:20});
    handler({clientX:50, clientY:60});
    console.log(JSON.stringify({rois, fetchBody}));
    """).replace('{handler}', handler)

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
