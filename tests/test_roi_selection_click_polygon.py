import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_click_creates_polygon_and_saves():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"frameContainer.addEventListener\('click',\s*\(e\) => \{([\s\S]*?drawAllRois\(\);\n\s*)\}\);", html)
    assert match, 'click handler not found'
    handler = match.group(1)

    script = textwrap.dedent("""
    let rois = [];
    let modules = ['m1'];
    let currentPoints = [];
    let drawingRect = false;
    let rectStart = null;
    let rectEnd = null;
    let hoverPoint = null;
    let currentSource = 'src';
    let fetchBody;
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
    handler({clientX:10, clientY:10});
    handler({clientX:50, clientY:10});
    handler({clientX:50, clientY:50});
    handler({clientX:10, clientY:50});
    console.log(JSON.stringify({rois, fetchBody}));
    """).replace('{handler}', handler)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout.strip())
    assert len(data['rois']) == 1
    pts = data['rois'][0]['points']
    assert pts[0] == {'x': 10, 'y': 10}
    assert pts[1] == {'x': 50, 'y': 10}
    assert pts[2] == {'x': 50, 'y': 50}
    assert pts[3] == {'x': 10, 'y': 50}
    payload = json.loads(data['fetchBody'])
    assert payload['rois'][0]['points'] == pts
