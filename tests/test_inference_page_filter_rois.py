import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_inference_page_filters_rois_by_group():
    html = Path('templates/partials/inference_page_content.html').read_text()
    match = re.search(r"function openRoiSocket\(\){[\s\S]*?};\n\s*}\n", html)
    assert match, 'openRoiSocket function not found'
    func_text = match.group(0)

    script = textwrap.dedent(f"""
        let rois = [];
        let allRois = [{{id:1, group:'A'}}, {{id:2, group:'B'}}];
        let pageNameEl = {{ innerText: '' }};
        let roiGrid = {{ innerHTML: '' }};
        const cam = 'x';
        const cellId = 'c1';
        global.location = {{ host: 'localhost' }};
        global.document = {{ getElementById: () => null }};
        function renderRoiPlaceholders(list=rois) {{ rois = list; }}
        global.WebSocket = function(url) {{ return {{}}; }};
        {func_text}
        openRoiSocket();
        roiSocket.onmessage({{ data: JSON.stringify({{group:'B'}}) }});
        console.log(pageNameEl.innerText + '|' + JSON.stringify(rois));
    """)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    page, rois_json = result.stdout.strip().split('|')
    assert page == 'B'
    assert json.loads(rois_json) == [{'id': 2, 'group': 'B'}]
