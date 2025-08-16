import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_update_page_roi_name_triggers_save():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"function updatePageRoiName\s*\(i, name\)\s*{[\s\S]*?}\n", html)
    assert match, 'updatePageRoiName function not found'
    func_text = match.group(0)

    script = textwrap.dedent(
        f"""
        let rois = [];
        let pageRois = [{{id:'p1', page:'old', name:'old', points:[{{x:1,y:2}}]}}];
        let otherRois = [];
        let currentSource = 'src';
        let fetchOpts;
        global.fetch = (url, opts) => {{ fetchOpts = opts; return Promise.resolve({{}}); }};
        {func_text}
        updatePageRoiName(0, 'new');
        console.log(fetchOpts.body);
        """
    )

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    body = json.loads(result.stdout.strip())
    assert body['rois'][0]['page'] == 'new'
    assert body['rois'][0]['name'] == 'new'
    assert body['source'] == 'src'
