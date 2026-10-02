#!/usr/bin/env python
"""Reprocesa imagenes puntuales desde backup_nohbg con un modelo mejor + alpha matting.

Uso: .venv/bin/python scripts/reprocess_bg.py birefnet-general "imgs/x.png" "sample_avatar.png"
"""
import os
import sys
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "backup_nohbg"


def main():
    model = sys.argv[1]
    files = sys.argv[2:]
    session = new_session(model, providers=["CPUExecutionProvider"])
    for rel in files:
        src = BACKUP / rel
        if not src.exists():
            # los .jpg originales se guardaron con su extension vieja
            for alt in src.with_suffix(".jpg"), src.with_suffix(".jpeg"):
                if alt.exists():
                    src = alt
                    break
        dst = ROOT / rel
        if not src.exists():
            print(f"NO HAY RESPALDO: {rel}")
            continue
        img = Image.open(src).convert("RGB")
        out = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=250,
            alpha_matting_background_threshold=5,
            alpha_matting_erode_size=8,
            post_process_mask=True,
        )
        out = Image.fromarray(out) if not isinstance(out, Image.Image) else out
        out = out.convert("RGBA")
        tmp = dst.with_suffix(dst.suffix + ".tmp.png")
        out.save(tmp)
        os.replace(tmp, dst)
        a = out.getchannel("A")
        print(f"{rel} {out.size} alfa={a.getextrema()}")


if __name__ == "__main__":
    main()
