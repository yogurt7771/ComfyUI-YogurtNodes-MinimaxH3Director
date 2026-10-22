# ComfyUI-YogurtNodes-MinimaxH3Director

**Yogurt H3 导演台**：一个节点，从可视化时间线直接出带声音的 MiniMax H3 成片。

在节点里点“打开导演台”，按镜头编排：每个镜头可以是文生、首尾帧、参考生成或源视频改写，并可叠加局部重绘（手绘、连线、文字识别或人脸遮罩，可放大贴回）；全片可开启二采放大。运行后成片保存到 `output/`，在节点中预览，`video` 输出可接 SaveVideo 或其他视频节点。

## 能力

| 能力 | 做法 |
| --- | --- |
| 文生 | 只写提示词；fl2va 原生 T2VA |
| 首尾帧 / 关键帧 | “首尾帧”页放首帧、尾帧（可只放尾帧）和任意中间帧；首尾帧按官方 FL2V 交给文本编码器（`<Picture 1>` / `<Picture 2>`），中间帧用 AddGuide 锚定。每张关键帧可选适配画幅：自动（官方：首帧拉伸、其余裁剪）/ 裁剪填满 / 补黑边 / 拉伸 |
| 参考生成 | “参考”页分全局参考和本镜参考：全局参考每一镜都带上，本镜参考只给这一镜；编号全局在前、本镜接着编（整体改写时源视频仍是 `<Video 1>`）。某一镜关掉“继承全局参考”就只用自己的。提示词里用 `<Picture N>` / `<Video N>` / `<Audio N>`，H3 建议每镜合计不超过 9 图 / 3 视频 / 3 音频。参考视频可设起点，点“截取”在素材条上选出参考的那一段。带参考的镜头用 `ref_model`（ref2va），其中的首尾帧作为固定画面锚点。每张参考图可选适配：原比例（官方，按 `ref_image_size` 等比缩小）/ 裁剪 / 补边 / 拉伸到成片画幅比例；参考图按成片尺寸取像素量，两段式二采的小首采也不会把它压小。每镜有“参考强度”（H3 的视觉条件强度，同时作用于这一镜的参考图、参考视频和关键帧，1 = 官方默认） |
| 源视频改写 | 源视频镜头：在舞台下方的素材条上拖动选区，从长素材里截出这一镜用的那一段（拖选区右缘改时长，或按 I / O 设入点 / 出点）。“保留原画”只改遮罩区，遮罩外像素原样输出；“整体改写”把源片段作为 `<Video 1>` 参考重新生成，可设改写强度（低于 1 时从源画面加噪起步） |
| 局部重绘 | 每层选遮罩来源：手绘（画笔 / 橡皮 / 矩形 / 反选）、连线遮罩（`masks` 输入，多帧逐帧对应）、文字识别（SAM 3 按英文短语在这一层的帧上跟踪目标，如 `girl in red raincoat`，逗号分隔多个，`cup:2` 最多取 2 个；中文目标在接了改写模型时自动翻成英文）、跟随人脸（YOLO 跟住框住的那张脸）、固定框。设帧范围、强度和单独提示词，可叠多层；“识别预览”只跑 SAM 3 / 检测（不加载 H3），把遮罩叠在画面上给你看。默认在首采上原位重绘，遮罩外 latent 保持不动，二采把它一起放大；接了 `inpaint_patch`（Fun ControlNet）时遮罩区更贴合周围画面，适合修补和去除，要在遮罩里加入新物体，不接时更听提示词。勾“放大贴回”则在二采后的成片上把遮罩区域裁出来、放大到设定像素量、只重绘遮罩内，再按羽化遮罩颜色匹配贴回，适合脸、手、小物体；原来的“局部精修”就是跟随人脸 / 固定框的放大贴回层，旧工程自动转换。节点参数 `detail` 是放大贴回总开关 |
| 镜头续接 | 勾选“接着上一镜”时，上一镜 AV latent 的最后 22 帧（节点参数 `overlap`，可选 5/39/56）拷入本镜开头并逐步释放（Drift-Control），画面和声音连续，重叠部分只输出一次 |
| 二采 | 开启后 `width` × `height` 是成片尺寸。两段式和低噪重绘可单开，也可同时开（先两段式、再低噪重绘），每一遍按自己的倍数放大后采样，首采（含局部重绘）在逐遍缩小后的小画幅上进行：例如 1344×768、`two_pass_scale` 2、`rerender_scale` 1.5 时是首采 448×256 → 两段式 896×512 → 低噪重绘 1344×768。两遍各有放大方式 `two_pass_upscaler` / `rerender_upscaler`：`h3_latent`（默认）用 H3 latent 放大模型直接放大视频 latent，不经过 VAE；`lanczos` / `model` 走 VAE 解码 → Lanczos 或放大模型 → VAE 编码（源视频局部重绘镜头要和源像素合成，总是走这条）。`two_pass`（两段式）：首采只跑调度表前面的步数，放大 `two_pass_scale` 倍后跑最后 `two_pass_steps` 步，总步数不变（如 8 = 6 + 2）；生成的声音沿首采时自己的噪声轨迹继续跑完，不重新加噪。`rerender`（低噪重绘）：放大 `rerender_scale` 倍到成片尺寸，从 `rerender_denoise` 的噪声再采 `rerender_steps` 步，音轨锁定；单开时接在跑完整个调度表的首采后，`rerender_scale` 为 1（默认）时不放大、直接在成片尺寸上重采。两遍各自缓存，只改低噪重绘的设置时不重跑两段式。续接镜头每一遍都接上一镜同一遍的 latent 尾部 |
| 声音 | 每镜可选：模型生成 / 配乐音轨（按时间线切分并锁定，口型跟随，成片保留原波形；在“镜头”页声音旁选择配乐，全片共用）/ 源视频原声（锁定）/ 静音 |
| 只重跑改动 | 每个阶段按内容缓存。改一个镜头只重跑它和接着它续拍的镜头；“换种子”重新抽这一镜或这一层 |

## 连接

| 输入 | 内容 |
| --- | --- |
| `model` | UNETLoader：`minimax_h3_fl2va_*`（文生、首尾帧、局部重绘） |
| `ref_model` | 可选，UNETLoader：`minimax_h3_ref2va_*`；带参考素材或整体改写源视频的镜头使用，不接则用 `model` |
| `clip` | CLIPLoader，type 选 `minimax`（Qwen3-VL） |
| `vae` / `audio_vae` | `minimax_h3_video_vae` / `minimax_h3_audio_vae` |
| `refine_model` | 可选，二采和放大贴回使用的模型（例如挂了放大 LoRA 的分支） |
| `rewriter` | 可选，Yogurt H3 改写模型：按剧情自动写每一镜的提示词（见下文），也把中文识别目标翻成英文 |
| `sam3` / `sam3_clip` | 可选，CheckpointLoaderSimple 加载 SAM 3 检查点（如 `sam3.1_multiplex_fp16.safetensors`）的 MODEL / CLIP；局部重绘的“文字识别”使用 |
| `inpaint_patch` | 可选，ModelPatchLoader：`MiniMax-H3-Fun-Controlnet-Union-2.0` |
| `upscale_model` | 可选，两段式或低噪重绘的放大方式选 `model` 时使用 |
| `sampler` / `sigmas` | 可选，替换 `sampler_name` / `steps`+`scheduler`（如 Turbo 采样器、ManualSigmas）；局部重绘按强度截取噪声表尾部 |
| `images` / `masks` / `videos` / `audios` | 可选，自动增长的图像 / 遮罩 / 视频（LoadVideo 等）/ 音频（LoadAudio 等）输入。编辑器里的素材位除了上传，也能在“连线素材…”里选对应类型的输入：图片位选图输入；视频位（源视频、参考视频）选视频输入或图像序列；音频位（参考音频、配乐）选音频输入或视频的原声。连线的源视频在运行前不能在舞台上预览 |

节点参数里是全片的渲染设置：`width` / `height`（成片画幅，32 的倍数；开二采时首采按各遍倍数缩小）、`overlap`（续接重叠帧）、`ref_image_size`（参考图尺寸）、`negative_prompt`、二采开关 `two_pass`（与 `two_pass_scale` / `two_pass_steps` / `two_pass_upscaler` / `two_pass_latent_model`）和 `rerender`（与 `rerender_scale` / `rerender_steps` / `rerender_denoise` / `rerender_upscaler` / `rerender_latent_model`）、`detail`（放大贴回总开关）与 `detail_steps`（放大贴回的步数）。导演台只管镜头、参考、声音和剧本。

`seed` 默认固定：固定种子才能复用已生成的镜头。H3 是蒸馏模型，`cfg` 通常保持 1；大于 1 时使用 `negative_prompt`。

重绘强度（含放大贴回）、低噪重绘强度和源视频改写强度都表示这一遍从多大的噪声开始（1 = 纯噪声）。步数保持设定值，噪声表按调度器在该噪声以下排布。H3 的 shift 为 12，按步数比例截取（BasicScheduler 的 denoise）会让 0.35 实际从 0.87 的噪声开始，几乎等于重新生成，所以这里不用那种算法。低噪重绘常用 0.25–0.4，放大贴回 0.3–0.5。`two_pass` 不用强度，从首采停下的噪声接着采：8 步、shift 6 时末 2 步约从 0.67 开始，shift 12 时约从 0.8 开始；起点越高越能补细节，也越可能改动长相，可减少 `two_pass_steps` 降低起点。

H3 latent 放大模型（[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)，约 690 MB）放在 `models/latent_upscale_models/`；`two_pass_latent_model` / `rerender_latent_model` 选默认文件而本地没有时，第一次用 `h3_latent` 二采会自动下载到这里（固定版本、校验 SHA-256；设置环境变量 `HF_ENDPOINT` 可走镜像）。

示例工作流：[example_workflows/yogurt_h3_director.json](example_workflows/yogurt_h3_director.json)。模型和素材需选择本机文件，插件不下载任何内容。

示例默认开启 Turbo 8 步，使用 [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) 的 768p 加速 LoRA，需放到 `models/loras/`：

| 分支 | LoRA | 采样设置 |
| --- | --- | --- |
| fl2va → `model` | `minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` | 后接 ModelSamplingMiniMaxH3，shift 6 / 3 |
| ref2va → `ref_model` | `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` | 不加 shift 节点（默认 12 / 3） |

导演台 `steps` 8、`euler`、`simple`、`cfg` 1；`width` × `height` 1344×768（LoRA 训练尺寸，更低分辨率下 768p LoRA 效果会变差）；`detail_steps` 3。二采默认关闭；开 `two_pass` 即首采 672×384 跑 6 步、成片 1344×768 跑最后 2 步（768p LoRA 在首采的小画幅上效果会差一些，细节靠最后两步补）。两个模型共用 `steps`，换 4 步时两个 LoRA 一起换成 4 步版（fl2v v1.3 / ref2v v1.0）。不用加速时把两个 LoRA 和 shift 节点旁路，`steps` 改回 25、采样器 `res_multistep`。

## 使用

1. 导入示例，选择本机模型。
2. 点“打开导演台”。底部时间线：“+ 生成镜头” / “+ 源视频镜头”，拖动镜头右边缘调整时长，斜纹表示续接的重叠帧。
3. 右侧按页编辑：剧本（剧情描述与自动改写）、镜头（时长、续接、提示词、声音与配乐）、首尾帧、参考、局部重绘。顶栏可导入 / 导出工程 JSON（镜头、参考和剧本；画幅、二采等在节点参数里）。
4. 舞台按时间线显示素材：源视频镜头显示截取的那段源视频，生成镜头按关键帧显示构图；空格从播放头一直播到全片结尾。运行后可在舞台工具栏切到“成片”。
5. “▶ 运行全片”。节点上的时间线显示每镜进度。
6. 要改哪里就改哪里，再运行：没变的镜头直接读缓存。生成镜头的局部重绘画在上次成片上。

快捷键：空格播放，←/→ 逐帧（Shift 一秒），绘制遮罩时 `[` `]` 调笔刷，Esc 退出绘制或关闭。

## 剧情改写（deepagents）

只写剧情也能出片：在“剧本”页写全局提示词，打开“自动改写提示词”，并把 **Yogurt H3 改写模型** 节点接到导演台的 `rewriter`。

- 全局提示词写整部片的情况（剧情、人物、风格、场景），是交给改写模型的背景，不会原样塞进每一镜；改写模型据此判断每一镜该拍什么。片段提示词是这一镜自己的要求，和全局提示词同时生效、在这一镜优先。某一镜关掉“继承全局提示词”，就只按它自己的片段提示词和时间线写。
- 不开自动改写时，全局提示词不参与生成，H3 只用每一镜的片段提示词。

- 改写由 [deepagents](https://github.com/langchain-ai/deepagents) 智能体完成，加载 MiniMax 官方 [h3-prompt-writing](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) 技能（随插件放在 `h3-prompt-writing/`）。智能体在内存中的虚拟文件系统里读技能和写作指南，不读写本机文件，也没有 shell。
- 智能体拿到的是导演台实际要生成的时间线：每镜的窗口时长、续接上一镜的重叠秒数、首尾帧和中间帧的对齐时间、参考素材标签（`<Picture N>` / `<Video N>` / `<Audio N>`）、声音锁定方式，以及每镜可选的“本镜要点”。它据此把剧情分到各镜，按 T2VA / I2VA / FL2VA / L2VA / Ref2VA 对应的官方格式写出完整提示词。智能体先列出每镜的分镜要点，再按时间顺序一镜一镜写、一镜一镜提交，写后一镜时前面各镜的提示词都在上下文里。开启“看图”时会把首尾帧、参考图和源视频（从入点那一帧）发给模型，全局参考图只发一次。不想让某张图离开本机时，在它下面取消“发给改写 AI 看”（参考图、参考视频、关键帧、源视频都有，默认勾选）：这张图不会发给模型，模型只知道这里有个 `<Picture N>` / `<Video N>`；可以另写一段文字描述代替它，不写就只按标签引用、不猜它长什么样。H3 生成不受影响。
- 每镜单独缓存。只有这一镜的设置变了，或它继承的全局提示词 / 全局要求变了才重写；续接镜头会随上一镜一起重写。“全部重新改写”换一版结果。
- 改写结果显示在节点面板（选中的镜头）、“镜头”页和“剧本”页；运行时每写完一镜就更新，不用等渲染结束。
- “生成剧本”只改写不渲染，结果与运行共用缓存。满意的某一镜可以“采用为手写”，之后就固定不再改写。
- 改写模型节点：`openai`（含 OpenAI 兼容接口，填 `base_url`）、`anthropic`、`ollama`。模型需要支持工具调用。`api_key` 留空时读取环境变量 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`（`base_url` 留空读 `OPENAI_BASE_URL`）；填写的 key 会保存在工作流里。`max_tokens` 默认 8192：OpenRouter 等服务在不设上限时会按模型最大输出预扣额度。OpenRouter 用 `openai` + `base_url=https://openrouter.ai/api/v1`，模型如 `anthropic/claude-opus-5.5`。
- 只有打开改写并运行，或点“生成剧本”时才会请求所选的模型服务。

安装：在 ComfyUI 的 Python 环境里 `pip install deepagents`。deepagents 声明依赖 `langchain-google-genai`，会把 `google-genai` 升到 2.x；若其他节点还需要 google-genai 1.x，可以改用：

```
python -m pip install "langchain>=1.4.6" "langchain-core>=1.6.11" "langchain-anthropic>=1.7.6" "langsmith>=0.14.11" "wcmatch>=11.0"
python -m pip install deepagents --no-deps
```

## 执行顺序

```
计划与缓存检查（只做需要的）
→ 文本与参考编码（CLIP + VAE）
→ 源视频镜头的识别遮罩（SAM 3 / 人脸检测）
→ 逐镜首采 / 编码源视频，原位局部重绘层（UNET，续接按顺序；生成镜头的识别遮罩随层在要重绘的画面上进行）
→ 二采，逐遍：全部放大到这一遍的画幅（latent 放大模型，或 VAE 解码放大编码）→ 全部采样（UNET；两段式跑调度表的最后几步，低噪重绘从低噪声再采一遍）
→ 逐镜解码、源视频合成（VAE）
→ 放大贴回的局部重绘层：全部识别、裁剪编码 → 全部采样 → 全部解码
→ 拼接成 MP4（H.264 + AAC）
```

同类模型的工作集中进行，避免大模型在每镜之间来回加载。缓存在 `output/yogurt_h3_director/cache/`：每镜每层的 latent、解码后的无损帧（FFV1）、生成音频、识别出的遮罩与放大贴回的裁剪。缓存不会自动删除，需要时手动清理该目录。

## 时长与续接

- H3 在 24fps、17k+5 帧网格上生成。镜头时长按网格向上对齐生成，输出时裁掉多余帧；后面接续拍镜头或设了尾帧的镜头输出完整窗口（最多多出 16 帧），时间线显示实际输出帧数。
- 续接镜头的首帧、关键帧、重绘层的帧号都从续接之后的第一帧算起。
- 源视频镜头按时间取帧换算到 24fps，任何帧率都可以；源视频镜头本身不续接上一镜，后面的镜头可以续接它。
- 上一镜太短（少于重叠帧）时自动改为硬切。

## 验证范围

`tests/` 用伪模型和真实媒体文件覆盖规划、遮罩网格、缓存复用与局部重跑、续接与二采尾部、源视频遮罩外像素保持、文字识别遮罩（用伪 SAM 3）、放大贴回只改遮罩内、识别预览、配乐锁定和输出帧数/时长。画质、SAM 3 与人脸跟踪效果和显存峰值需用真实模型检查。

```
python -B -m unittest discover -s custom_nodes/ComfyUI-YogurtNodes-MinimaxH3Director/tests -t custom_nodes/ComfyUI-YogurtNodes-MinimaxH3Director/tests
node --check custom_nodes/ComfyUI-YogurtNodes-MinimaxH3Director/web/editor.js
```

`h3-prompt-writing/` 是 MiniMax 官方的 H3 提示词写作指南，遵循该目录内的 MiniMax H3 Community License。

## 许可

GPL-3.0。出处见 [NOTICE.md](NOTICE.md)。
