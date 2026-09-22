# ComfyUI-YogurtNodes-MinimaxH3Director

MiniMax H3 分段导演插件，两种规划、一个采样：

- 角色替换：视频＋角色图 → 长视频分段导演台 → 计划。
- 多段故事：角色卡＋剧情＋段数 → 故事分段规划 → 计划。
- 长视频采样：只接模型、采样器、sigmas 和计划；逐段生成，上一段 AV latent 尾部直接续接下一段开头，可选放大二采，去重叠后合成 24fps 视频。角色替换计划恢复源音轨，故事计划使用生成音频。

计划携带分辨率、分段、提示词、参考图、源视频、参考组和精修设置，采样节点不再单独设置这些输入。

独立安装，不依赖也不修改其它 Director、TimelineDirector 或 ComfyUI 核心。

## 使用

重启 ComfyUI、刷新网页，导入 [长视频角色替换示例](example_workflows/yogurt_h3_long_video_character_replace.json)。上传视频与角色图，填写已有 Yogurt OpenRouter 节点的密钥，修改独立的“每段秒数”整数节点（默认 6），点击导演台内“刷新计划”，检查后运行。首采分辨率由导演台的 `megapixels` 决定（默认 0.41，16:9 源片为 864×480），画面比例跟随源视频。模型文件仍从本机选择，不自动下载。

图片识别只描述看得见的外观，不根据“韩立”等名字补全身份。示例使用 [OpenRouter Gemini 3.8 Flash](https://openrouter.ai/google/gemini-3.8-flash)，亦可换支持图片输入的模型；本插件不读取或保存 API Key。已有识别结果可以直接接入提示词节点，完全跳过 OpenRouter。

## 已实现能力

| 能力 | 使用方式 |
| --- | --- |
| 自动有限分段 | 按源时长、独立 INT 秒数和 H3 帧网格计算，包含尾段 |
| 跨段续接 | overlap_frames 默认 22；上一段采样后的 AV latent 尾部直接拷入下一段开头，输出去除重复头部 |
| 可视时间线 | 源视频预览、点击定位、斜纹重叠区、拖动右边界或输入窗口帧数、切点、逐段状态 |
| 逐段编辑 | 完整提示词覆盖、目标人物描述、角色索引、种子偏移、保留原片 |
| 镜头切分 | scene_detection 开启缩略灰度画面差分检测；阈值可调，亦可手动插入/移除切点 |
| 断点续跑 | 每段成功后原子写入无损 safetensors，重启后 cache_mode=reuse 恢复 |
| 局部重做 | 时间线“标记本段重做”，或 rerun_segments 填 2,4-6；正常点击运行 |
| 分段结果预览 | 已完成/已缓存段可无声预览；包含该段重叠头部，最终视频不会重复 |
| 多角色 | 导演台 additional_roles 接附加角色图；时间线角色 0 为主图，1 起对应附加图 |
| 多模态参考 | “多模态参考组”接到规划节点：附加图片、24fps 视频帧、配对音轨或独立音频 |
| 精修二采 | “精修与放大”接到规划节点；Lanczos、超分模型或 RTX VSR 放大后低去噪二次采样，可选另一套兼容 H3 模型 |
| 通用模式 | “通用导演条件”支持原生 T2V、I2V、首尾帧和 R2V，输出条件与 latent，接标准采样节点 |
| 时间线复用 | 导入/导出 JSON；工作流本身亦保存逐段编辑 |

## 连续性与时长

输出帧数为 round(源秒数 × 24)，精度为一帧。6 秒对应的常规 H3 窗口是 141 帧（5.875 秒）；22 帧重叠时，后续窗口通常新增 119 帧。末尾不足 5 帧的新内容会吸收入前一窗口，生成时再向上补齐至 17k+5，最后去掉补帧。窗口秒数是规划目标，不是绝对生成时长上限。

重叠支持 1 帧或 5/22/39/56… 帧，输入自动向下适配；0 关闭。5 帧以上时，上一段**已经生成的** AV latent 尾部直接拷入下一段开头，去噪遮罩让它保持干净并随 sigma 调度逐步释放；1 帧时用上一段末帧作 AddGuide 引导。它不是把两个独立结果淡入淡出，但也不能保证每次无缝或身份绝对不漂移。

latent 尾部只能在 H3 的 17k+5 帧网格上对齐，所以后面还接有续接段的窗口长度会向下对齐到 17k+5（如时间线填 240 帧实际为 226 帧）；末段不受影响。

镜头切点不跨镜头重叠；角色索引变化或保留原片的边界不续接，仍按绝对时间去重。源音轨只解码一次，以绝对采样点对齐输出；SaveVideo 可能重新编码音频，并非原容器音频的 bitstream copy。

## 缓存与重做

缓存目录：ComfyUI/output/yogurt_h3_cache。每段保存完整有效窗口的浮点画面、续接所需的 AV latent（开启二采时包含首采和二采两份）以及故事计划的生成音频；磁盘开销较大，不自动删除。角色替换和多段故事都支持续跑与局部重做。

- reuse：复用匹配结果；refresh：重新生成全部；off：不读写磁盘缓存。
- 素材内容、上游模型/采样配置、角色图、提示词、参考媒体、种子、精修设置、前段 latent 尾部等变化会影响键。文件模型以路径和大小/修改时间标识；原地替换且保留时间戳或修改自定义补丁时，手动改变 cache_revision。
- 重做某段若生成尾帧变化，后续依赖它的段会自动失效。这是连续性依赖，不应强制拼上旧下游片段。
- 完成局部重做后清空 rerun_segments，否则下次仍会重做所选段。
- 只在整段成功后替换检查点；中断的段重新生成，之前成功的段可复用。

## 时间线编辑与参考编号

逐段提示词为空时继承全局；填写后替换整个提示词。切点是源视频 24fps 时间轴上的绝对帧号。窗口帧数 0 表示继承全局，24 帧为 1 秒。拖动边界后重新计算后续窗口。

编辑以 start_frame 匹配。调整边界后不再匹配的旧编辑保留在导出 JSON 中并提示，**不会静默套用到其他片段**。更换源视频后需要“重建时间线”，该按钮会确认清除当前切点和逐段编辑，不删除视频或缓存。

直接连接 LoadVideo 时支持运行前视频预览；其他 VIDEO 来源可正常生成并在执行后显示计划。自动切镜为轻量画面差分，闪光/快速运动可能误检，需检查切点。

角色替换中 Picture1 为本段角色图、Video1 为本段源片；附加参考从 2 开始。配对音轨 video_audio_N 对应 video_N，稀疏连接也保持配对。附加视频需先转换为 24fps 帧批次。原生文本条件引用这些素材，不等于自动遮罩或多人跟踪：复杂多人场景需写清目标并检查结果。

## 精修与依赖

基础链仅使用当前 ComfyUI 的 MiniMax H3 原生节点和已有依赖。自动角色图识别示例另需 ComfyUI-YogurtNodes。

精修节点接在规划节点的 `refine` 上，未连接时不会放大或二次采样。`megapixels` 是二采输出的百万像素数，画面比例跟随计划。放大方式：

- lanczos：本地常规缩放，无额外依赖。
- model：连接 UpscaleModelLoader；复用 ImageUpscaleWithModel。
- nvidia_rtx_vsr：复用 KJNodes 的 ImageResizeKJv2，需该节点及兼容的 nvidia-vfx/NVIDIA 环境。本插件不安装 SDK，不在失败时偷偷降级。
- resample：放大后 VAEEncode，用原生 Separate/Concat AV Latent 接回首采音频，在全分辨率重新编码条件（参考图、首尾帧），再用 `BasicScheduler(steps, denoise)` 做低去噪 H3 采样；采样器、种子、提示词与首采相同。续接段的首采接上一段首采的 latent 尾部，二采接上一段二采的尾部。第二模型必须兼容当前视频/音频 VAE。

[精修示例](example_workflows/yogurt_h3_character_replace_refine.json)额外连接精修节点；[通用条件示例](example_workflows/yogurt_h3_native_modes.json)展示标准采样链。通用条件示例是单段生成。

## 多段故事长视频

导入 [多段故事示例](example_workflows/yogurt_h3_story_long_video.json)。

| 节点 | 作用 |
| --- | --- |
| Yogurt H3 故事分段规划 | 剧情、模式（T2VA/I2VA/FL2VA/L2VA/Ref2VA）、段数、每段秒数、重叠帧、画面比例、megapixels、角色名/备注、角色卡图片、参考组、精修 → 计划与给 LLM 的改写请求 |
| Yogurt H3 解析 LLM 分段 | 用 LLM 返回的 JSON 替换计划中的提示词，段数与时长保持不变；失败时沿用模板规划 |
| Yogurt H3 选取分段提示词 | 取出计划中某段完整提示词，用于预览或单段生成 |
| Yogurt H3 角色卡叠图 | 角色卡叠成 batch，给 Generate Text 看图改写 |
| Yogurt H3 长视频采样 | 与角色替换共用：按计划逐段生成、续接、去重叠、合成视频 |

角色卡图片接到规划节点的 `cards`，顺序即 `<Picture 1>`、`<Picture 2>`…。Ref2VA 把角色卡作为参考图（其后接多模态参考组的图片）；I2VA 首段用卡 1 作首帧，L2VA 末段用卡 1 作尾帧，FL2VA 用卡 1/卡 2 作首段首帧/末段尾帧。

`continue_audio` 开启时，重叠区的音频 latent 一起续接，末 8 个 token 用半余弦释放。合并时，重叠画面归前一段，重叠音频归后一段。所有段使用同一种子。

首采分辨率由 `aspect_ratio` 和 `megapixels` 共同决定，宽高各自对齐到 32。默认 1.03 MP：16:9 → 1344×768，9:16 → 768×1344，1:1 → 1024×1024。

输出帧数 = 段数 × 每段帧数 − (段数 − 1) × 重叠帧；每段帧数为每段秒数向上对齐到 17k+5。

## 提示词与官方指南

提示词按 MiniMax 官方 [h3-prompt-writing](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) 指南的结构书写：基础模式为关键帧对齐句 + `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`，Ref2VA 为六段式。每段都是完整独立的提示词，不再拼接全局提示词。

- **Yogurt H3 提示词改写**：连接本地文本生成模型（Qwen 系列 CLIPLoader），输入粗略需求、模式、时长和参考素材说明，输出一段 H3 提示词和 report。image/video/audio 可让模型看到参考素材。
- **Yogurt H3 故事分段改写**：接故事计划，用本地模型一次改写全部分段，段数与时长不变；旁路时直接使用模板提示词。
- 每段都是独立生成的视频，视频模型只读到本段提示词：提示词只描述画面与声音，不提及其他段；段间衔接由采样时的 latent 局部重绘完成，第 k 段开头的画面描述与第 k−1 段结尾一致。短故事会按叙事弧（开端→结局）扩写成指定段数，每段的镜头编号与时间都从本段 00:00 起算。
- `standalone_check`（默认开）：改写后让同一模型逐段做语义检查，找出只有看过其他段才讲得通的说法并替换为直接描述；subject_definitions 与 retention_analysis 不改动，改了什么写入 report。
- 两个本地节点都用 JSON 约束解码：键名和 JSON 结构由节点写入，字符串内只屏蔽会破坏 JSON 的 token，因此输出总能解析、每个段落都有内容；段落标题、官方顺序和关键帧对齐句由节点按字段拼出。只支持字节级 BPE 的模型（Qwen 系列）；实现上在本次生成期间临时包装文本模型的 `sample_token`。被 `max_length` 截断时保留已写内容并在 report 中说明。
- 外部 LLM（如 OpenRouter）使用规划节点的 `llm_prompt`，按同一套字段 JSON 输出，再接“解析 LLM 分段”回填；缺段落的分段沿用模板并写入 parse_report。
- 官方指南原样放在 `h3-prompt-writing/`，运行时读取，遵循其自带的 MiniMax H3 Community License（见该目录 LICENSE / NOTICE），不属于本插件的 GPL-3.0。缺少该目录时相关节点会报错并给出下载地址。

## 验证边界

测试覆盖规划、音画区间、latent 尾部拷贝与遮罩、Drift-Control 调度、缓存失效、续跑/重做、模式与参考配对、图展开和二采 latent 连接；不等同于完整模型画质验收。自动切镜效果、RTX VSR 和真实长视频角色一致性仍取决于素材、模型与运行环境。

当前合并输出完整 IMAGE 批次，长视频最终拼接需要较多系统内存；磁盘检查点用于续跑，不代表无限长度流式渲染。未承诺像素级多人隔离、自动跟踪遮罩或所有场景无缝。

## 开发验证

使用已安装 ComfyUI 依赖的 Python，运行：

```
python -B -m unittest discover -s custom_nodes/ComfyUI-YogurtNodes-MinimaxH3Director/tests -v
node --check custom_nodes/ComfyUI-YogurtNodes-MinimaxH3Director/web/director.js
```

## 许可

GPL-3.0。段间 latent 续接代码改编自 GPL-3.0 项目，出处见 [NOTICE.md](NOTICE.md)。
