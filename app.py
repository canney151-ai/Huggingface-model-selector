import os
import json
import time
import threading
import uuid
import re
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

import requests

app = Flask(__name__)

CONFIG_FILE = Path.home() / ".config" / "hf-gguf-dl" / "config.json"
DEFAULT_MODELS_DIR = Path.home() / "models"
MAX_FILE_SIZE_BYTES = 88 * 1024 * 1024 * 1024  # 88 GB — leaves ~8 GB headroom on 96 GB

HF_API = "https://huggingface.co/api"
HF_BASE = "https://huggingface.co"

downloads: dict = {}
downloads_lock = threading.Lock()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"models_dir": str(DEFAULT_MODELS_DIR), "hf_token": ""}


def save_config(config: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def models_dir() -> Path:
    return Path(load_config().get("models_dir", DEFAULT_MODELS_DIR))


def hf_headers() -> dict:
    token = load_config().get("hf_token", "")
    h = {"User-Agent": "hf-gguf-downloader/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/settings", methods=["GET"])
def get_settings():
    cfg = load_config()
    # Don't expose full token, just whether one is set
    cfg["hf_token_set"] = bool(cfg.get("hf_token"))
    return jsonify(cfg)


@app.route("/api/settings", methods=["POST"])
def update_settings():
    cfg = load_config()
    data = request.json or {}
    if "models_dir" in data:
        cfg["models_dir"] = data["models_dir"]
    if "hf_token" in data:
        cfg["hf_token"] = data["hf_token"]
    save_config(cfg)
    Path(cfg["models_dir"]).mkdir(parents=True, exist_ok=True)
    return jsonify({"status": "ok"})


@app.route("/api/models")
def search_models():
    query = request.args.get("q", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    limit = 24
    offset = (page - 1) * limit

    params: dict = {
        "filter": "gguf",
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
        "offset": offset,
    }
    if query:
        params["search"] = query

    try:
        resp = requests.get(
            f"{HF_API}/models",
            params=params,
            headers=hf_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()

        models = []
        for m in raw:
            models.append({
                "id": m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "lastModified": m.get("lastModified", ""),
                "tags": [t for t in m.get("tags", []) if not t.startswith("arxiv:")],
            })

        return jsonify({"models": models, "page": page, "query": query, "has_more": len(raw) == limit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def parse_quantization(filename: str) -> str:
    name = filename.upper()
    for pattern in [r"(IQ\d+_[A-Z0-9]+)", r"(Q\d+_K_[MS])", r"(Q\d+_K\b)", r"(Q\d+_\d)", r"\b(F16|BF16|F32)\b"]:
        m = re.search(pattern, name)
        if m:
            return m.group(1)
    return "Unknown"


RECOMMENDED_QUANTS = {"Q4_K_M", "Q5_K_M", "Q4_K_S", "IQ4_XS", "Q6_K", "Q5_K_S"}
QUALITY_ORDER = ["IQ3_XS", "IQ4_XS", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S",
                 "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32"]


def quant_sort_key(quant: str) -> int:
    try:
        return QUALITY_ORDER.index(quant)
    except ValueError:
        return 99


@app.route("/api/models/<path:model_id>/files")
def get_model_files(model_id):
    try:
        resp = requests.get(
            f"{HF_API}/models/{model_id}",
            params={"blobs": "true"},
            headers=hf_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        files = []
        for s in data.get("siblings", []):
            fname = s.get("rfilename", "")
            if not fname.lower().endswith(".gguf"):
                continue

            size = s.get("size") or 0
            if size and size > MAX_FILE_SIZE_BYTES:
                continue  # Won't fit in 96 GB

            quant = parse_quantization(fname)
            files.append({
                "filename": fname,
                "size": size,
                "size_gb": round(size / (1024 ** 3), 2) if size else None,
                "quantization": quant,
                "recommended": quant in RECOMMENDED_QUANTS,
                "sort_key": quant_sort_key(quant),
            })

        files.sort(key=lambda x: (not x["recommended"], x["sort_key"], x["size"] or 0))

        return jsonify({
            "id": model_id,
            "files": files,
            "model_card": data.get("cardData", {}),
            "total_filtered": len(files),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download", methods=["POST"])
def start_download():
    data = request.json or {}
    model_id = data.get("model_id", "").strip()
    filename = data.get("filename", "").strip()

    if not model_id or not filename:
        return jsonify({"error": "Missing model_id or filename"}), 400

    # Security: local filename is basename only — no path traversal possible
    local_name = Path(filename).name
    if not local_name or ".." in local_name:
        return jsonify({"error": "Invalid filename"}), 400

    dest = models_dir() / local_name
    if dest.exists():
        return jsonify({"error": "already_exists", "path": str(dest)}), 409

    task_id = str(uuid.uuid4())
    with downloads_lock:
        downloads[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "total": 0,
            "speed_bps": 0,
            "filename": local_name,
            "model_id": model_id,
            "path": str(dest),
            "error": None,
            "started_at": time.time(),
        }

    t = threading.Thread(target=_download_worker, args=(task_id, model_id, filename, dest), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})


def _download_worker(task_id: str, model_id: str, filename: str, dest: Path):
    url = f"{HF_BASE}/{model_id}/resolve/main/{filename}"
    tmp = dest.with_suffix(".gguf.tmp")

    def _update(**kw):
        with downloads_lock:
            downloads[task_id].update(kw)

    try:
        _update(status="connecting")
        with requests.get(url, headers=hf_headers(), stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            _update(status="downloading", total=total)

            downloaded = 0
            last_time = time.time()
            last_bytes = 0
            chunk = 1024 * 512  # 512 KB

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for data in resp.iter_content(chunk_size=chunk):
                    if data:
                        f.write(data)
                        downloaded += len(data)
                        now = time.time()
                        elapsed = now - last_time
                        if elapsed >= 1.0:
                            speed = (downloaded - last_bytes) / elapsed
                            last_time = now
                            last_bytes = downloaded
                            _update(progress=downloaded, speed_bps=speed)
                        else:
                            _update(progress=downloaded)

        tmp.rename(dest)
        _update(status="complete", progress=downloaded, total=downloaded, speed_bps=0)

    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        _update(status="error", error=str(e))


@app.route("/api/download/progress/<task_id>")
def download_progress(task_id):
    def generate():
        while True:
            with downloads_lock:
                info = downloads.get(task_id)
            if info is None:
                yield f"data: {json.dumps({'error': 'not_found'})}\n\n"
                break
            yield f"data: {json.dumps(info)}\n\n"
            if info["status"] in ("complete", "error"):
                break
            time.sleep(0.75)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/downloads")
def list_downloads():
    with downloads_lock:
        return jsonify(list(downloads.values()))


@app.route("/api/downloaded")
def list_downloaded():
    md = models_dir()
    if not md.exists():
        return jsonify({"files": [], "models_dir": str(md)})

    files = []
    for f in sorted(md.glob("*.gguf")):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size": stat.st_size,
            "size_gb": round(stat.st_size / (1024 ** 3), 2),
            "path": str(f),
            "quantization": parse_quantization(f.name),
            "modified": stat.st_mtime,
        })

    return jsonify({"files": files, "models_dir": str(md)})


@app.route("/api/downloaded/<path:filename>", methods=["DELETE"])
def delete_model(filename):
    if "/" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    fpath = models_dir() / filename
    if not fpath.exists():
        return jsonify({"error": "not_found"}), 404
    fpath.unlink()
    return jsonify({"status": "deleted"})


@app.route("/api/download/<task_id>", methods=["DELETE"])
def cancel_download(task_id):
    with downloads_lock:
        info = downloads.get(task_id)
    if not info:
        return jsonify({"error": "not_found"}), 404
    # Mark cancelled; the worker will exit on next write error
    with downloads_lock:
        downloads[task_id]["status"] = "cancelled"
    # Remove tmp file if present
    tmp = Path(info["path"]).with_suffix(".gguf.tmp")
    tmp.unlink(missing_ok=True)
    return jsonify({"status": "cancelled"})


if __name__ == "__main__":
    models_dir().mkdir(parents=True, exist_ok=True)
    print(f"  Models directory : {models_dir()}")
    print(f"  Open             : http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
