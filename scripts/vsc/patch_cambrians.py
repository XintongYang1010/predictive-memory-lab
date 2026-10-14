#!/usr/bin/env python3
"""Apply the V6.1 logging patch to a cambrian-s checkout at 058b1b4.

  python scripts/vsc/patch_cambrians.py /path/to/cambrian-s

Changes, all inert unless PMR_VSC_DUMP is set:
  * tasks/cambrians_vsc/utils.py: cache dir name from CAMBRIANS_VSC_CACHE_NAME
    (default unchanged), so repeat/phase/reverse caches reuse the task;
  * models/simple/cambrians_vsc.py: per frame log the memoryless surprise,
    the pooled SigLIP2 feature and the LFP prediction; per segment log its
    frame range and decoded count; write one .npz per video.
The official boundaries, LFP head, threshold and summation are untouched.
Both target files are checked against their sha256 at 058b1b4 first.
"""
import hashlib
import sys
from pathlib import Path

MODEL = "lmms-eval/lmms_eval/models/simple/cambrians_vsc.py"
UTILS = "lmms-eval/lmms_eval/tasks/cambrians_vsc/utils.py"
SHA = {MODEL: "43b7c19418ba68ac03b294cede8c6cd24ac57c37085742bc03d65650c51eca19",
       UTILS: "98980e82e2c5f649aa269b06b38d0056874ad091a9dda7abdef8d3d8d0dbaa4b"}
MARK = "# PMR-V6.1 logging patch"

HELPER = '''

''' + MARK + '''
def _pmr_dump(log, visual_tensor_paths, to_float, doc_id):
    import numpy as _np
    d = os.environ["PMR_VSC_DUMP"]
    os.makedirs(d, exist_ok=True)
    name = str(visual_tensor_paths)
    name = name[:-3] if name.endswith(".pt") else name
    cache = os.environ.get("CAMBRIANS_VSC_CACHE_NAME", "cambrians_vsc")
    key = name.split("_" + cache + "_", 1)[-1]
    C = len(log["d"][0])
    p = _np.stack([x if x is not None else _np.full(C, _np.nan, dtype="float16") for x in log["p"]])
    _np.savez(os.path.join(d, key + ".npz"), s=_np.asarray(log["s"], dtype="float32"), d_pool=_np.stack(log["d"]),
              p_pool=p, seg_frames=_np.asarray(log["seg"], dtype="int64").reshape(-1, 2),
              seg_counts=_np.asarray([to_float(o) for o in log["out"]], dtype="float64"),
              seg_text=_np.asarray(log["out"]), doc_id=_np.asarray(doc_id), video_key=_np.asarray(key))
'''

EDITS = [
    ("from lmms_eval.api.registry import register_model\n",
     "from lmms_eval.api.registry import register_model\n" + HELPER),
    ("                outputs = []\n",
     "                outputs = []\n"
     "                _pmr_log = {\"s\": [], \"d\": [], \"p\": [], \"seg\": [], \"out\": []} "
     "if os.environ.get(\"PMR_VSC_DUMP\") else None  " + MARK + "\n"),
    ("                    frame_feature = add_newline_tokens(frame_feature).unsqueeze(0)\n",
     "                    if _pmr_log is not None:  " + MARK + "\n"
     "                        _pmr_log[\"s\"].append(float(surprisingness_score))\n"
     "                        _pmr_log[\"d\"].append(vit_visual_features[frame_idx].float().mean((0, 1)).cpu().numpy().astype(\"float16\"))\n"
     "                        _pmr_log[\"p\"].append(frame_feature_prediction.float().mean((1, 2))[0].cpu().numpy().astype(\"float16\") if frame_idx > 0 else None)\n"
     "                    frame_feature = add_newline_tokens(frame_feature).unsqueeze(0)\n"),
    ("                        episodic_starts = episodic_ends\n",
     "                        if _pmr_log is not None:  " + MARK + "\n"
     "                            _pmr_log[\"seg\"].append((episodic_starts - 1, episodic_ends - 1))\n"
     "                        episodic_starts = episodic_ends\n"),
    ("                        outputs.append(self.tokenizer.decode(output_ids[0, 1:].tolist()))\n",
     "                        outputs.append(self.tokenizer.decode(output_ids[0, 1:].tolist()))\n"
     "                        if _pmr_log is not None:  " + MARK + "\n"
     "                            _pmr_log[\"out\"].append(outputs[-1])\n"),
    ("            outputs = str(sum([to_float(output) for output in outputs]))\n",
     "            if _pmr_log is not None:  " + MARK + "\n"
     "                _pmr_dump(_pmr_log, visual_tensor_paths, to_float, doc_id)\n"
     "            outputs = str(sum([to_float(output) for output in outputs]))\n"),
]
UTILS_EDIT = ('cache_name = "cambrians_vsc"\n',
              'cache_name = os.getenv("CAMBRIANS_VSC_CACHE_NAME", "cambrians_vsc")  ' + MARK + "\n")


def main(root):
    root = Path(root)
    for rel in (MODEL, UTILS):
        txt = (root / rel).read_text()
        if MARK in txt:
            print(f"already patched: {rel}")
            continue
        got = hashlib.sha256(txt.encode()).hexdigest()
        if got != SHA[rel]:
            raise SystemExit(f"{rel}: sha256 {got} != pinned {SHA[rel]} (not cambrian-s @058b1b4?)")
        edits = EDITS if rel == MODEL else [UTILS_EDIT]
        for old, new in edits:
            if txt.count(old) != 1:
                raise SystemExit(f"{rel}: anchor not unique: {old!r}")
            txt = txt.replace(old, new)
        (root / rel).write_text(txt)
        print(f"patched: {rel}")


if __name__ == "__main__":
    main(sys.argv[1])
