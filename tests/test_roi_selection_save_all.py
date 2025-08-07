import json
import re
import subprocess
import textwrap
from pathlib import Path

def test_save_all_rois_payload():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"(function saveAllRois\s*\(\)\s*{[\s\S]*?})\n\s*function clearAllRois", html)
    assert match, 'saveAllRois function not found'
    func_text = match.group(1)

    script = textwrap.dedent(
        """
        let rois = [
          {id:'1', module:'m1', points:[{x:1,y:2}]},
          {id:'2', module:'m2', points:[{x:3,y:4}]}
        ];
        let currentSource = 'src';
        let fetchBody;
        global.fetch = (url, opts) => { fetchBody = opts.body; return Promise.resolve({json: () => Promise.resolve({filename:'f'}), ok:true}); };
        global.alert = () => {};
        global.confirm = () => true;
        function loadRois(){}
        {func}
        saveAllRois();
        console.log(fetchBody);
        """
    ).replace('{func}', func_text)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    body = json.loads(result.stdout.strip())
    assert 'rois' in body
    assert len(body['rois']) == 2
    for roi in body['rois']:
        assert set(['id', 'module', 'points']).issubset(roi.keys())
