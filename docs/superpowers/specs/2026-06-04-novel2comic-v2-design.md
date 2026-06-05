# Novel2Comic V2 系统设计文档

> 日期: 2026-06-04 | 状态: Draft | 小说→漫画智能生成系统

## 一、项目目标

构建一个"小说→漫画"智能生成系统。输入任意长度的小说章节文本，经过分析、角色设计、场景拆分、分镜、生图、排版六个阶段，最终输出带对话框和特效字的完整漫画图片。

核心差异化能力：
- **多风格自动适配**：根据小说题材自动判断漫画风格（日式Manga/韩式Webtoon/中式古风），并预留 Skill 扩展机制
- **交互式协作**：用户作为"导演"，在 Pipeline 各阶段通过自然语言介入调整
- **角色一致性**：四层防线确保同一角色在所有格子中外观一致
- **人物关系图谱**：知识图谱建模角色间的情感关系，自动影响分镜构图
- **反馈记忆自优化**：三层记忆体系让系统越用越懂用户偏好

## 二、整体架构

采用 **混合架构**: Pipeline 骨架 + Agent 交互层 + 共享数据总线。

```
📖 小说章节输入
      │
      ▼
┌─────────────────────────────────┐
│  🧠 Agent 交互层（用户导演协作）    │
│  自然语言 → Pipeline 操作翻译      │
│  审核/微调/重做三种协作模式        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ⚙️ Pipeline 执行引擎（6 阶段）    │
│  ① 文本分析 → ② 角色设计           │
│  → ③ 场景拆分 → ④ 分镜生成         │
│  → ⑤ 图像生成 → ⑥ 漫画排版          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  🗄️ 共享数据总线                  │
│  项目配置/角色库/场景列表/分镜脚本    │
│  已生成图片/版本历史/关系图谱       │
└─────────────────────────────────┘
```

### 三层职责

| 层 | 做什么 | 不做什么 |
|---|---|---|
| Agent 交互层 | 接收用户自然语言意图 → 翻译为 Pipeline 操作 → 展示结果 → 收集反馈 | 不直接生成内容、不直接调生图 API |
| Pipeline 引擎 | 6 阶段顺序执行，每阶段有明确的输入/输出 Schema，可独立测试和替换 | 不做用户交互、不做自然语言理解 |
| 数据总线 | 持有全流程状态，支持序列化/反序列化，提供版本历史和快照 | 不包含业务逻辑 |

## 三、风格系统

### 3.1 风格判断流程

文本分析阶段提取特征词 → 风格分类器映射到 StyleProfile → Pipeline 各阶段读取 StyleProfile 约束输出。

分析维度：题材标签、叙事节奏、文化背景、语言风格、情感基调。

### 3.2 三种内置风格

| 风格 | 色彩 | 阅读方向 | 画幅 | SD 关键词 |
|------|------|----------|------|-----------|
| 🎌 Manga | 黑白+灰度网点 | 右翻页 | B5 (~1.45:1) | `manga style, black and white, screentone, speed lines, line art` |
| 🇰🇷 Webtoon | 全彩色柔和调色板 | 竖屏滑动 | 9:16 | `webtoon style, full color, soft palette, manhwa, vertical scroll` |
| 🏮 古风 | 水墨风/工笔重彩 | 灵活 | 9:16 或 1:1 | `chinese ink painting style, gufeng, watercolor wash, ancient chinese comic` |

### 3.3 风格映射规则

- 武侠/仙侠/玄幻 → 中式古风
- 轻小说/校园/恋爱 → Manga 日式
- 都市/职场/现实 → Webtoon 韩式
- 科幻/悬疑 → 根据节奏自动判断
- 历史/古装 → 中式古风

### 3.4 风格扩展机制

每种风格定义为 StyleProfile 对象，包含约 15-20 个可覆盖参数。后期可通过 Skill 文件扩展新风格，Skill 机制提供风格约束的 YAML+Markdown 描述，可插拔组合。

## 四、Pipeline 6 阶段详解

### ① 文本分析（LLM 驱动）

- **输入**: 小说章节全文
- **输出**: `{ genre_tags, style, tone, era, pace, characters_preview }`
- **用户介入 🎯**: 确认/修改风格判断、补充世界观设定、标注重要人物
- **策略**: 短章(≤3000字)直接全文分析；长章先分段摘要再汇总

### ② 角色设计（LLM + 生图）

- **输入**: 分析结果中的人物列表 + 原文外貌描写段落
- **输出**: `CharacterSheet[] { name, role, appearance, reference_image_url, sd_trigger_words, personality_notes }`
- **用户介入 🎯**: 审核外貌描述 → 生成定妆参考图 → 不满意可重绘 → 确认锁定
- **策略**: 首次出场角色从原文提取外貌，返回角色从 CharacterSheet 复用

### ③ 场景拆分（LLM 驱动）

- **输入**: 原文 + 分析结果 + 角色列表
- **输出**: `Scene[] { id, title, summary, characters_in_scene, emotion_arc, key_dialogue, suggested_panels }`
- **用户介入 🎯**: 审查场景列表：合并/拆分/删减场景、调整每场景格数
- **策略**: 按地点/时间/情绪变化切分，一章通常 3-8 个场景

### ④ 分镜生成（LLM 驱动 · 核心阶段）

- **输入**: 场景 + 角色定妆信息 + StyleProfile
- **输出**: `Panel[] { panel_number, visual_description, character_action, dialogue, camera_angle, mood, sd_prompt, character_refs }`
- **用户介入 🎯**: 逐格审查画面描述、调镜头角度、改台词、调整 sd_prompt、增删分镜格
- **策略**: 逐场景生成，相邻格间强制景别/视角变化。sd_prompt 自动注入 StyleProfile 基座 + 角色触发词 + 画幅比例
- **质量规范**:
  - 每场景 3-6 格分镜，动作场景可到 8 格
  - 画面描述必须有构图信息（前景/中景/背景）
  - 关键情感转折台词不能遗漏
  - 人物首次出现描述外貌特征，后续用名字指代
  - 相邻格之间要有视觉变化

### ⑤ 图像生成（云端 API）

- **输入**: 每格 Panel 的 sd_prompt + CharacterSheet 参考图 + StyleProfile
- **输出**: `GeneratedImage[] { panel_id, image_url, seed, model_used, consistency_score }`
- **用户介入 🎯**: 逐格查看生图结果 → 标记满意/不满意 → 重绘不满意的格子
- **策略**: 首次取每格 1 张候选。用户确认后可一键批量出图，也可逐格精细调整

### ⑥ 漫画排版（渲染引擎）

- **输入**: 已生成图片 + 分镜脚本 + StyleProfile 排版规则
- **输出**: `ComicPage[] { page_number, panel_layout, speech_bubbles, sfx_overlays, final_image }`
- **用户介入 🎯**: 预览排版 → 调整对话框位置/样式 → 添加特效字 → 确认导出
- **策略**: Manga 用格阵排版（支持异形格、出血），Webtoon/古风用纵向拼接。对话框根据 StyleProfile 选择气泡样式，自动计算放置位置

## 五、角色一致性方案

四层防线从生成前、生成中、生成后锁定角色一致性：

### 第一层（🔴 生成前）：Character Sheet 定妆
每个角色生成 1 张定妆照（正面半身、中性表情、完整服饰、纯色背景），用户审核通过后锁定写入角色库。

### 第二层（🟡 生成时）：Prompt 约束
每个角色的 `sd_trigger_words` 自动注入所有该角色出现的格 prompt 中。

### 第三层（🔵 生成时）：Reference Image 传参
根据云平台能力选择策略：IP-Adapter → Reference-only → img2img → 纯 prompt 兜底。

### 第四层（🟢 生成后）：一致性检测
CLIP/ViT 嵌入余弦相似度比对，低于阈值(0.75)的自动标记 ⚠️，提示用户确认或自动重绘。

### 特殊场景处理
- **新角色中途登场**: 追加到角色库，不影响前面已生成的图片
- **角色外貌变化**（受伤/变装/年龄变化）: 创建角色"变体"，从原角色继承基础外貌
- **远景/背影/剪影**: 跳过人脸相似度检查，只验证服饰颜色/体型
- **群像场景**: 使用更强的模型 + 每角色一致性期望降低 + 用户手动审核权重提高

## 六、人物关系知识图谱

### 6.1 数据模型

```
CharacterGraph {
  nodes: CharacterNode[]      // 角色节点
  edges: RelationshipEdge[]   // 关系边
  timeline: RelationEvent[]   // 关系变化事件
}

CharacterNode {
  id, name, role_type,        // "protagonist" | "antagonist" | ...
  faction,                    // 所属势力/阵营
  importance: 1-10,           // 影响出场频率和特写优先级
  first_appearance_chapter,
  status                      // "active" | "dead" | "missing" | "unknown"
}

RelationshipEdge {
  from, to,
  relation_type,              // "血缘" | "爱情" | "友情" | "敌对" | "师徒" | "主仆" | "利用"
  sub_type,                   // "暗恋" | "杀父之仇" | "青梅竹马" | "背叛"
  intimacy: -10 ~ +10,        // -10(不共戴天) ~ +10(生死相依)
  power_dynamic,              // "平等" | "A主导" | "B主导" | "互相制衡"
  public_knowledge,           // 关系是否公开
  current_tension,            // "和谐" | "紧张" | "暧昧" | "一触即发"
  shared_history              // 共同经历
}

RelationEvent {               // 追踪关系变化
  timestamp, chapter,
  edge: (from, to),
  change: { field, old_value, new_value, trigger_event }
}
```

### 6.2 图谱自动影响分镜

| 关系模式 | 自动影响的分镜参数 |
|----------|-------------------|
| intimacy ≥ +7（生死相依） | 双人中近景、眼神交流、柔和光线、sd_prompt 追加 "close together, warm atmosphere" |
| intimacy ≤ -7（不共戴天） | 斜线对峙构图、低角度仰拍、特写眼神交锋、sd_prompt 追加 "tense standoff, dramatic shadows" |
| power_dynamic "A主导" | A 仰拍显高大、B 俯拍显弱小、A 占更大画面比例 |
| public_knowledge = false | 公开场合站远、表情刻意、只在对视瞬间流露微表情 |
| current_tension "暧昧" | 避免直视、侧脸和偷看视角、sd_prompt 追加 "shy glance, soft focus" |

### 6.3 图谱更新流程

每章完成后，LLM 自动提取本章中的人物关系变化 → 生成图谱更新提案 → 用户确认后写入。第一版用 JSON 内嵌存储（节点≤50），预留图数据库升级路径。

## 七、反馈记忆与自优化系统

### 7.1 三层记忆架构

| 层级 | 内容 | 生命周期 |
|------|------|----------|
| 📁 项目级 | 角色外貌修正、场景风格偏好、台词润色历史、被拒绝的 prompt | 项目期间 → 完成后归档 |
| 👤 用户级 | 镜头偏好、节奏偏好、色调偏好、对话框风格、忌讳清单 | 永久 → 持续进化 |
| 🎨 风格级 | 古风墨色浓淡/manga网点密度/webtoon色彩饱和度、SD关键词效果评分 | 永久 → 跨项目积累 |

### 7.2 反馈学习闭环

用户修改 → LLM 提取偏好向量 → 写入对应记忆层 → 自动注入后续 Pipeline 阶段

### 7.3 各阶段记忆注入

- ① 文本分析：注入用户历史风格判断修正
- ② 角色设计：注入角色外貌修正历史 + 角色设计偏好
- ④ 分镜：注入镜头偏好 + 节奏偏好 + 场景特殊要求
- ⑤ 生图：注入 SD 关键词效果评分 + 角色触发词修正 + 忌讳清单
- ⑥ 排版：注入对话框大小/位置偏好 + 排版间距偏好

### 7.4 技术实现

第一版：JSON 文件 + LLM 规则提取（从反馈文本中提取结构化偏好）。后续可升级为向量数据库（ChromaDB/Qdrant）做相似场景自动检索。

## 八、交互模型

### 8.1 三种协作模式

- **👁️ 审核模式**: Pipeline 每阶段完成暂停，展示结果等用户确认
- **🔧 微调模式**: 针对单个格子或元素做精准调整，不影响其他内容
- **🔄 重做模式**: 对不满意的结果整体重来，但保留已有偏好信息

### 8.2 自然语言指令示例

| 用户说 | Agent 翻译 | Pipeline 操作 |
|--------|-----------|---------------|
| "整体风格改成日式热血漫" | 切换 StyleProfile | 更新数据总线 → 触发分镜+生图重做 |
| "第一格加个全景" | 场景1 插入新 Panel | 插入 → 重编号 → 只生新格 |
| "苏墨在暗巷的镜头太亮了" | 定位指定格 → 调暗 sd_prompt | 更新 prompt → 重绘 → 保留其他格 |
| "回到分镜阶段" | 回退到 Stage ③ | 恢复快照 → ④⑤⑥ 标记 outdated |
| "全部确认，下一章" | 保存 + 加载新文本 | 角色库持久化 → 记忆持久化 → 新 Pipeline |

### 8.3 版本历史与回退

- **数据总线快照**: 每个 Pipeline 阶段完成后自动创建
- **操作日志**: 记录所有用户操作和 Pipeline 动作，支持 undo/redo
- **分支实验**: 允许从任意快照分叉，对比不同方案

### 8.4 交互界面

第一版：CLI 自然语言对话模式。后续可升级为 Web UI（图片预览+并排对比）或 IDE 插件。

## 九、技术选型

| 模块 | 推荐方案 | 理由 |
|------|----------|------|
| 🖥️ 运行环境 | Python 3.10+ | LLM SDK、图像处理库生态最成熟 |
| 🧠 LLM | DeepSeek / Claude / GPT-4o | 长上下文 + 结构化输出。通过 Adapter 切换 |
| 🖼️ 生图 API | Stability AI / Replicate / 即梦 | 优先选支持 Reference Image 的 API，通过 Adapter 切换 |
| 👤 角色一致性 | IP-Adapter (via Replicate) | 根据云平台能力选用，无参考能力的降级为纯 prompt |
| 🔍 一致性检测 | CLIP / DINOv2 嵌入 + 余弦相似度 | 轻量级，CPU 可运行 |
| 🎨 排版渲染 | Pillow (PIL) + 自建排版引擎 | 初期够用，后期可换 ReportLab/Skia |
| 💾 数据持久化 | JSON 文件 + 项目目录 | 人可读、可 git 版本控制 |
| 🧠 反馈记忆 | JSON profile + LLM 规则提取 | 第一版不引入向量数据库 |
| 🕸️ 关系图谱 | JSON 内嵌（预留图DB升级路径） | 节点≤50 时完全够用 |

## 十、项目结构

```
novel2comic/
├── src/
│   ├── agent/                      # Agent 交互层
│   ├── pipeline/                   # 6 阶段 Pipeline 引擎
│   │   ├── stage1_analyze.py
│   │   ├── stage2_characters.py
│   │   ├── stage3_scenes.py
│   │   ├── stage4_storyboard.py
│   │   ├── stage5_image_gen.py
│   │   └── stage6_layout.py
│   ├── databus/                    # 数据模型 + 序列化
│   ├── memory/                     # 反馈记忆系统
│   ├── graph/                      # 人物关系知识图谱
│   ├── styles/                     # StyleProfile 定义（可扩展为 Skill）
│   └── adapters/                   # LLM + ImageGen API 适配层
├── skills/                         # 风格/分镜 Skill 文件（.md）
├── projects/                       # 用户项目存储
│   └── {project_name}/
│       ├── project.json            # 项目配置 + 角色库 + 图谱 + 完整数据总线
│       ├── user_profile.json       # 用户级偏好（跨项目共享）
│       ├── style_memory.json       # 风格级经验积累
│       ├── snapshots/              # 阶段快照
│       └── outputs/                # 生成的图片 + 漫画
└── requirements.txt
```

## 十一、核心数据模型（概要）

项目以 `Project` 为顶层实体，包含 StyleProfile、CharacterSheet[]、Chapter[]、CharacterGraph、UserProfile、StyleMemory。

Chapter 下挂 Analysis → Scene[] → Panel[] → GeneratedImage → ComicPage[]。

所有实体通过数据总线共享，支持 JSON 序列化和快照恢复。

## 十二、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 云端 API 角色参考图能力不足 | 角色一致性下降 | 强化第二层 prompt 约束 + 第四层检测重绘兜底 |
| 长章节上下文超限 | 分析/分镜质量下降 | 分段摘要策略 + 关键信息压缩 |
| 生图 API 成本高 | 单章成本不可控 | 首次只生 1 张候选 + 用户确认后再批量 |
| 用户审美主观性强 | 一致性检测不适用 | 以用户手动审核为主，自动检测只做辅助提醒 |
