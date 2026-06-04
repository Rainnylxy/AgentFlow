# Novel2Comic Agent 设计文档

> 日期: 2026-06-04 | 状态: Draft | 独立项目（不修改 AgentFlow 代码）

## 一、目标

基于 AgentFlow 框架（ToolKit + Memory + Prompt + Thinking Engine），构建一个"小说→漫画分镜"Agent。输入任意长度的小说文本，输出标准漫画分镜脚本 + 每格对应的 SD/图片生成 prompt。

## 二、项目结构

```
novel2comic/
  agent.py           ← Agent 定义 + 4 个 Tool + 入口
  requirements.txt   ← agentflow + openai
  example.py         ← 示例运行脚本
  outputs/           ← 生成的分镜脚本输出目录
  .env.example       ← API key 配置模板
```

不修改 `agentflow/` 下的任何代码。

## 三、架构

```
用户输入小说文本
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Agent (ThinkingMode.ADAPTIVE)                   │
│                                                  │
│  Prompt: 漫画分镜师模板                           │
│    - RoleCard: 专业漫画分镜师                     │
│    - StyleGuide: manga/webtoon 风格规范          │
│    - OutputFormat: 分镜字段定义                  │
│    - QualityRules: 质量约束                       │
│                                                  │
│  Tools:                                          │
│    analyze_text(text) → 文本分析                 │
│    extract_scenes(text, max_scenes) → 场景拆分   │
│    storyboard_scene(scene, chars, style) → 分镜  │
│    compile_chapter(title, scenes, style) → 汇总  │
│                                                  │
│  Memory:                                         │
│    Episodic → 记住人物设定/关系                   │
│    Semantic → 缓存风格参考                        │
│                                                  │
│  Thinking: AdaptiveRouter 自动选策略              │
│    短文本 → ReAct                                │
│    长文本 → PlanExecute                          │
└─────────────────────────────────────────────────┘
      │
      ▼
  Markdown 输出（分镜表 + SD prompt 列表）
```

## 四、4 个 Tool 定义

### Tool 1: analyze_text

```
输入: text: str
输出: {type, style, characters: [{name, role, traits}], tone, era}
功能: 分析文本类型、风格、人物、基调
```

### Tool 2: extract_scenes

```
输入: text: str, max_scenes: int = 8
输出: [{id, title, summary, characters_involved, emotion, key_dialogue}]
功能: 拆分为关键场景列表
```

### Tool 3: storyboard_scene（核心）

```
输入: scene_summary, characters: [{name,role,traits}], style(manga/webtoon/auto), panels_per_scene=4
输出: [{panel_number, visual_description, character_action, dialogue, camera_angle, mood, sd_prompt}]
功能: 为单个场景生成分镜
```

### Tool 4: compile_chapter

```
输入: chapter_title, scenes_storyboard: [{scene_id, panels, ...}], style
输出: str (Markdown 格式)
功能: 汇总为最终输出
```

## 五、Prompt 模板

### Section: RoleCard
"你是一位专业的漫画分镜师(Comic Storyboard Artist)，精通日本漫画(ネーム/Name)和韩式条漫(Webtoon)的分镜设计。你的任务是将小说文字转化为可视化的漫画分镜脚本。"

### Section: StyleGuide
- manga: 黑白、右翻页、视觉动线引导、特写与远景交替、速度线与集中线、对话框融入构图
- webtoon: 彩色、竖屏滑动、每格宽度一致、人物居中偏上、对话框在上方、留白控制阅读节奏
- auto: 轻小说→manga，网文→webtoon，现代都市→webtoon，武侠玄幻→manga

### Section: OutputFormat
每格分镜字段: panel_number, visual_description(中文), character_action, dialogue, camera_angle, mood, sd_prompt(英文)

### Section: QualityRules
1. 每场景 3-6 格，不要过多或过少
2. 画面描述含构图信息: 前景/中景/背景
3. sd_prompt 包含: anime style/manga style, 画幅比例, 色彩提示, 关键元素
4. 关键对话不能遗漏
5. 人物首次出现时描述外貌特征，后续用名字指代

## 六、思考模式

使用 `ThinkingMode.ADAPTIVE`，由 AdaptiveRouter 自动选择:
- 短文本(≤ 一段文字) → ReAct: 直接分析→拆场景→分镜一气呵成
- 长文本(一整章) → PlanExecute: 先制定分镜计划→逐场景执行→汇总
- 需要自我修正 → ReflectionWrapper 自动包裹

## 七、记忆策略

使用 `MemoryProfile.standard()`:
- Working: 当前场景的上下文
- Episodic: 跨场景记住人物设定、关系、已出现过的外貌描述
- 检索门: 新场景自动检索已有人物信息，避免重复介绍

## 八、AgentBuilder 伪代码

```python
agent = (AgentBuilder("novel2comic")
    .with_llm(OpenAIClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    ))
    .with_tools(analyze_text, extract_scenes, storyboard_scene, compile_chapter)
    .with_prompt(COMIC_PROMPT_TEMPLATE)
    .with_memory(MemoryProfile.standard())  # 记住角色设定
    .with_thinking(ThinkingMode.ADAPTIVE)
    .with_max_iterations(20)  # 长的多步任务需要更多迭代
    .build())
```
