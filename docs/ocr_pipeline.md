# OCR Pipeline Architecture

This document summarizes how OCR work is dispatched for each region of interest (ROI) in a frame and the safeguards in place for performance and memory usage.

## ROI Extraction and Group Filtering
* `process_frame` filters ROIs by the active group and ensures each ROI defines four points before processing.
* Perspective warp is executed in a worker thread via `asyncio.to_thread` so that the event loop remains responsive, and the warped ROI is copied to avoid sharing the original frame buffer.
* When a warp cannot be produced (degenerate points or transform failure) an axis-aligned crop of the frame is used as a fallback, and finally a tiny placeholder tile is generated as a last resort. Each ROI therefore always produces an image for OCR and a result record every frame without stalling the pipeline.

## Inference Scheduling
* Each scheduled ROI bundles the callable, arguments, and callback future into the global `_INFERENCE_QUEUE`.
* The queue capacity is `MAX_WORKERS * 10` where `MAX_WORKERS` equals `os.cpu_count() or 1`, ensuring enough buffering without unbounded growth.
* `_EXECUTOR` spins up `MAX_WORKERS` threads, each running `_inference_worker`, so CPU-bound OCR modules such as RapidOCR fully utilize available cores while preventing oversubscription.

## Result Handling and Cleanup
* When an inference future completes, `_on_done` encodes optional ROI thumbnails in JPEG with quality 80 using worker threads, minimizing memory spikes from large raw arrays. Results emitted with a fallback extraction are tagged with `extraction_mode` so downstream consumers can monitor degraded cases.
* Pending inference bookkeeping and timeouts ensure stale jobs are flushed, keeping latency low even when some modules are slow.

Overall, this design provides concurrent OCR processing across ROIs, leverages the CPU efficiently, and limits memory pressure by copying only the necessary warped ROI regions and compressing optional artifacts before transmission.
