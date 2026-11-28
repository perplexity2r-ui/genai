#!/usr/bin/env python3
"""Скачивает int8-модель Depth Anything V2 ViT-S (27 МБ) в models/.

Почему именно эта модель (замеры на 2 ядрах CPU, 720p кадр):

    MiDaS small   (65 МБ, 256x256)   ~120 мс  — старее, хуже на тонких скалах
    DA V2 ViT-S   (99 МБ fp32)       ~290 мс  — качество лучше, но медленно
    DA V2 ViT-S int8 (27 МБ)         ~140 мс  — то же качество на глаз, 2x быстрее

int8-квантование даёт 27 МБ и ~2x скорость при почти том же качестве — это и есть
«маленькая, быстрая и качественная» для 12 FPS на CPU.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

# Основное зеркало — GitHub API (raw.githubusercontent.com часто блокируется
# TLS-фильтрами), запасное — обычный raw-URL.
MIRRORS = (
    "https://api.github.com/repos/MartinDelophy/depth-anything-quantize/"
    "contents/models/depth_anything_vits14.quant.onnx",
    "https://raw.githubusercontent.com/MartinDelophy/depth-anything-quantize/"
    "master/models/depth_anything_vits14.quant.onnx",
)
URL = MIRRORS[0]
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "da_vits14_int8.onnx")
MIN_BYTES = 20_000_000


def _progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    bar = "#" * int(pct / 2)
    sys.stdout.write(f"\r  [{bar:<50s}] {pct:5.1f}%  {done / 1e6:5.1f}/{total / 1e6:.1f} МБ")
    sys.stdout.flush()


def download_any(mirrors=MIRRORS, dest: str = DEST) -> str:
    """Пробует зеркала по очереди; возвращает путь к модели."""
    errors = []
    for url in mirrors:
        try:
            return download(url, dest)
        except Exception as exc:  # noqa: BLE001 - нужен перебор зеркал
            errors.append(f"{url}: {exc}")
            print(f"  не удалось: {exc}")
    raise RuntimeError("не удалось скачать модель ни с одного зеркала:\n" + "\n".join(errors))


def download(url: str = URL, dest: str = DEST) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"скачивание: {url}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "genshin-parkour/0.1",
        "Accept": "application/vnd.github.raw, application/octet-stream, */*",
    })
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            _progress(done, total)
    print()
    size = os.path.getsize(tmp)
    if size < MIN_BYTES:
        os.unlink(tmp)
        raise RuntimeError(f"файл слишком мал ({size} байт) — возможно, это не модель")
    os.replace(tmp, dest)
    sha = hashlib.sha256(open(dest, "rb").read()).hexdigest()[:16]
    print(f"готово: {dest}  ({size / 1e6:.1f} МБ, sha256:{sha})")
    return dest


if __name__ == "__main__":
    if len(sys.argv) > 1:
        download(sys.argv[1])
    else:
        download_any()
