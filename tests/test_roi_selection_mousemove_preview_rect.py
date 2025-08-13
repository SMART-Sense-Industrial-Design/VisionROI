import json
import re
import subprocess
import textwrap
from pathlib import Path


def test_mousemove_updates_hover_point_for_rectangle():
    html = Path('templates/roi_selection.html').read_text()
    match = re.search(r"frameContainer.addEventListener\('mousemove',\s*\(e\) => \{([\s\S]*?)\}\);", html)
    assert match, 'mousemove handler not found'
    handler = match.group(1)

    script = textwrap.dedent("""
    let hoverPoint = null;
    let currentPoints = [];
    let drawingRect = true;
    let rectStart = {x:10, y:20};
    const frameContainer = { getBoundingClientRect: () => ({left:0, top:0, width:100, height:100}) };
    const canvas = { width:100, height:100 };
    function drawAllRois(){}
    function handler(e){
    {handler}
    }
    handler({clientX:50, clientY:60});
    console.log(JSON.stringify({hoverPoint}));
    """).replace('{handler}', handler)

    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout.strip())
    assert data['hoverPoint'] == {'x': 50, 'y': 60}
