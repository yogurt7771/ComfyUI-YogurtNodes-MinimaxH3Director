# Notice

Copyright (c) 2026 Yogurt. Distributed under the GNU General Public License v3.0 (see LICENSE).

- `continuation.py` is adapted from
  [Songssx/ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector)
  (GPL-3.0): finite direct-latent continuation, Drift-Control AV masks and the soft audio seam.
  Drift-Control AV originates from ethanfel/ComfyUI-MiniMaxH3-Contex-Loop (GPL-3.0).
- `detail.py` follows the tracking and stitching approach of ComfyUI-H3-FaceRefine (MIT, Carasibana)
  as integrated in [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
  (Apache-2.0): per-frame detection, gap interpolation, smoothed trajectory, affine crop, colour-matched feathered paste.
- The director layout (shot timeline, first/last-frame, reference and source-video shots, refine and face
  passes) takes its workflow from ComfyUI_MiniMaxH3_Director and the TimelineDirector material timeline.
- `rewrite.py` runs MiniMax's `h3-prompt-writing` skill (from [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills))
  with [deepagents](https://github.com/langchain-ai/deepagents) (MIT), installed separately by the user.
- `h3-prompt-writing/` is MiniMax's H3 prompt-writing guide under the MiniMax H3 Community License
  (see that folder's LICENSE / NOTICE); it is not covered by this plugin's GPL-3.0.

Conditioning uses ComfyUI's native MiniMax H3 implementation; no third-party source or weights are bundled.
