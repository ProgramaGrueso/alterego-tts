#!/usr/bin/env python
"""Quita el fondo de todas las imagenes del proyecto (recursivo, Solo imagenes de la raiz).

- Respalda cada original en backup_nohbg/<ruta_relativa> una sola vez.
- Reescribe el archivo original con canal alfa transparente.
"""
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "backup_nohbg"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MODEL = sys.argv[1] if len(sys.argv) > 1 else "u2net"
ALPHA_MATTING = "--matting" in sys.argv


def targets():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if ".tmp." in p.name:
            continue
        rel = p.relative_to(ROOT)
        parts = rel.parts
        if parts[0] in {".venv", ".git", "backup_nohbg", "liveportrait_src"}:
            continue
        yield p, rel


def main():
    providers = os.environ.get("REMBG_PROVIDERS", "CPUExecutionProvider").split(",")
    session = new_session(MODEL, providers=providers)
    todo = list(targets())
    print(f"{len(todo)} imagenes | modelo={MODEL} | matting={ALPHA_MATTING}")
    for i, (path, rel) in enumerate(todo, 1):
        bak = BACKUP / rel
        if not bak.exists():
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak)
        img = Image.open(path)
        if img.mode == "RGBA" and img.getchannel("A").getextrema()[0] == 0:
            print(f"[{i}/{len(todo)}] {rel} ya tiene fondo quitado, se omite")
            continue
        img = img.convert("RGB")
        out = remove(
            img,
            session=session,
            alpha_matting=ALPHA_MATTING,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=6,
            post_process_mask=True,
        )
        out = Image.fromarray(out) if not isinstance(out, Image.Image) else out
        out = out.convert("RGBA")
        # sanity: hay transparencia real
        alpha = out.getchannel("A")
        lo, hi = alpha.getextrema()
        hist = alpha.histogram()
        total = alpha.width * alpha.height
        pct_transp = sum(hist[:16]) / total
        tmp = path.with_suffix(path.suffix + ".tmp.png")
        out.save(tmp)
        os.replace(tmp, path)
        print(
            f"[{i}/{len(todo)}] {rel} {out.size} alfa=({lo},{hi}) "
            f"transp={pct_transp * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
