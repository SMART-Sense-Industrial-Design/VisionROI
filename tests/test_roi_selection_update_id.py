import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_update_roi_id_triggers_save():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"function updateRoiId\s*\(i, id\)\s*{[\s\S]*?}\n", html)
    assert match, 'updateRoiId function not found'
    func_text = match.group(0)

    script = textwrap.dedent(
        f"""
        let rois = [{{id:'1', group:'g', module:'m', points:[{{x:1,y:2}}]}}];
        let pageRois = [];
        let otherRois = [];
        let currentSource = 'src';
        let fetchOpts;
        function drawAllRois() {{}}
        global.fetch = (url, opts) => {{ fetchOpts = opts; return Promise.resolve({{}}); }};
        {func_text}
        updateRoiId(0, '2');
        console.log(fetchOpts.body);
        """
    )

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    body = json.loads(result.stdout.strip())
    assert body['rois'][0]['id'] == '2'
    assert body['source'] == 'src'

