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

- `latent_upscale.py` ports the LatentResizer3D network of
  [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  (MIT, notice below). Its weights are not bundled: they are downloaded on first use from
  [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) into `models/latent_upscale_models`.

Conditioning uses ComfyUI's native MiniMax H3 implementation; no third-party weights are bundled.

```
MIT License

Copyright (c) 2026 LBH-123-AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
