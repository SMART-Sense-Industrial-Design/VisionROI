import asyncio
from .stubs import stub_quart

stub_quart()
from app import app


def test_inference_routes():
    async def run():
        client = app.test_client()
        resp = await client.get('/inference')
        assert resp.status_code == 200
        resp = await client.get('/inference_page')
        assert resp.status_code == 200

    asyncio.run(run())
