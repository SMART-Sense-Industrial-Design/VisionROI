import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_delete_page_roi_removes_and_saves():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"function deletePageRoi\s*\(i\)\s*{[\s\S]*?}\n", html)
    assert match, 'deletePageRoi function not found'
    func_text = match.group(0)

    script = textwrap.dedent("""
        let rois = [];
        let pageRois = [{id:'1', points:[{x:1, y:2}]}];
        let otherRois = [];
        let currentSource = 'src';
        function drawAllRois(){};
        function renderRoiList(){};
        let fetchOpts;
        let confirmCalls = [];
        global.confirm = msg => { confirmCalls.push(msg); return true; };
        global.fetch = (url, opts) => { fetchOpts = opts; return Promise.resolve({json: () => Promise.resolve({filename:'f'})}); };
        __FUNC_TEXT__
        deletePageRoi(0);
        console.log(JSON.stringify({pageRois, fetchOpts, confirmCalls}));
        """).replace('__FUNC_TEXT__', func_text)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout.strip())
    assert data['pageRois'] == []
    body = json.loads(data['fetchOpts']['body'])
    assert body['rois'] == []
    assert body['source'] == 'src'
    assert data['confirmCalls'] == ['Delete this Page ROI?']

