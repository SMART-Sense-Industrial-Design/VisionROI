import pytest
from app import app

@pytest.mark.asyncio
async def test_inference_routes():
    client = app.test_client()
    resp = await client.get('/inference')
    assert resp.status_code == 200
    resp = await client.get('/inference_page')
    assert resp.status_code == 200
