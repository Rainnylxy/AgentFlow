"""AgentFlow CLI — 入口命令"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(help="AgentFlow - 生产级多 Agent 编排与评测框架")
console = Console()

# ---------------------------------------------------------------------------
# 模板内容
# ---------------------------------------------------------------------------

SAMPLE_WORKFLOW_YAML = """\
name: hello-world
description: "一个简单的示例 Workflow：问候 → 处理 → 回复"
global_timeout_ms: 120000
nodes:
  - id: greet
    kind: agent
    label: "问候"
    agent:
      model: gpt-4o
      prompt: "你是一个友好的助手"
      memory_scope: inherit
  - id: process
    kind: agent
    label: "处理"
    agent:
      model: gpt-4o
      prompt: "你负责处理用户的请求"
      memory_scope: inherit
    timeout_ms: 60000
  - id: reply
    kind: agent
    label: "回复"
    agent:
      model: gpt-4o
      prompt: "你负责生成最终回复"
      memory_scope: workflow
edges:
  - from: greet
    to: process
  - from: process
    to: reply
"""

SAMPLE_SKILL_MD = """---
name: customer-support
description: "客户支持能力"
tools: [knowledge_base_search]
---

# 客户支持

你是专业客服助手，帮助用户解决问题。

## 流程
1. 理解用户的请求
2. 查询知识库获取相关信息
3. 给出清晰准确的回复

## 约束
- 不要编造信息
- 不确定时告知用户需要进一步确认
- 保持礼貌和专业

## 输出格式
用简洁清晰的语言回复用户，必要时使用列表或步骤说明。
"""

SAMPLE_EVAL_YAML = """\
name: basic-eval
cases:
  - id: greeting
    input: "你好"
    expected: "你好"
  - id: math
    input: "1+1是多少？"
    expected: "2"
"""

SAMPLE_CONFIG_YAML = """\
# AgentFlow 项目配置
name: hello-world
version: "0.1.0"

# 默认 LLM 配置
llm:
  provider: openai
  model: gpt-4o
  base_url: https://api.openai.com/v1
  # api_key 从环境变量 AGENTFLOW_API_KEY 读取

# 运行时配置
runtime:
  max_iterations: 10
  thinking_mode: adaptive
  memory_profile: standard

# 工具注册
tools: []
  # - name: search
  #   kind: local
  #   path: "my_project.tools:search"
  #   description: "搜索知识库"

# Skill 目录
skills_dir: skills/

# Eval 目录
evals_dir: evals/
"""


# ---------------------------------------------------------------------------
# new — 脚手架
# ---------------------------------------------------------------------------

@app.command()
def new(
    name: str = typer.Argument(..., help="项目名称"),
    dir: str = typer.Option(".", help="创建目录"),
):
    """创建一个新的 AgentFlow 项目。"""
    project_dir = Path(dir) / name

    if project_dir.exists():
        console.print(f"[red]X 目录已存在: {project_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[green][+] 创建 AgentFlow 项目: {name}[/green]")

    # 目录结构
    (project_dir / "workflows").mkdir(parents=True)
    (project_dir / "skills").mkdir(parents=True)
    (project_dir / "evals").mkdir(parents=True)

    # 文件
    (project_dir / "workflows" / "hello.yaml").write_text(SAMPLE_WORKFLOW_YAML, encoding="utf-8")
    (project_dir / "skills" / "customer-support.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
    (project_dir / "evals" / "basic.yaml").write_text(SAMPLE_EVAL_YAML, encoding="utf-8")
    (project_dir / "agentflow.yaml").write_text(SAMPLE_CONFIG_YAML, encoding="utf-8")

    console.print(f"[green]OK 项目已创建: {project_dir}[/green]")
    console.print()
    console.print("  [dim]├── workflows/hello.yaml[/dim]")
    console.print("  [dim]├── skills/customer-support.md[/dim]")
    console.print("  [dim]├── evals/basic.yaml[/dim]")
    console.print("  [dim]└── agentflow.yaml[/dim]")
    console.print()
    console.print("[yellow]tip: 下一步:[/yellow]")
    console.print(f"    cd {project_dir}")
    console.print(f"    agentflow dev workflows/hello.yaml")


# ---------------------------------------------------------------------------
# dev — 开发运行
# ---------------------------------------------------------------------------

@app.command()
def dev(
    workflow_path: str = typer.Argument("workflow.yaml", help="Workflow YAML 文件路径"),
    llm_model: str = typer.Option("", help="使用的 LLM 模型（覆盖 agentflow.yaml 配置）"),
    dry_run: bool = typer.Option(False, help="干跑模式——不调 LLM，只看 DAG 执行"),
):
    """启动开发服务器，运行 Workflow 并查看 Trace。"""
    from agentflow.dsl.serializer import from_yaml
    from agentflow.dsl.types import NodeKind
    from agentflow.runtime.orchestrator import DAGExecutor
    from agentflow.runtime.thinking import ThinkingMode
    from agentflow.runtime.config import ProjectConfig

    # 加载项目配置 (agentflow.yaml)
    wf_path = Path(workflow_path)
    config_path = wf_path.parent / "agentflow.yaml"
    if config_path.exists():
        config = ProjectConfig.load(config_path)
        console.print(f"[dim]已加载项目配置: {config_path}[/dim]")
    else:
        config = ProjectConfig._default(wf_path.parent)
        console.print("[dim]未找到 agentflow.yaml，使用默认配置[/dim]")

    # 加载 Workflow
    console.print(f"[blue]>> 加载 Workflow: {workflow_path}[/blue]")
    try:
        wf = from_yaml(workflow_path)
    except FileNotFoundError:
        console.print(f"[red]X 文件不存在: {workflow_path}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]X 解析失败: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]OK Workflow '{wf.name}' 加载成功 ({len(wf.nodes)} 节点, {len(wf.edges)} 边)[/green]")

    # Mermaid 可视化
    from agentflow.dsl.visualizer import to_mermaid
    console.print()
    console.print(Panel(to_mermaid(wf), title="DAG 可视化", border_style="dim"))

    # 确定 LLM 配置
    model = llm_model or config.llm.model
    api_key = config.llm.resolved_api_key
    base_url = config.llm.resolved_base_url

    # 构建 node_id → Agent 映射
    agents: dict[str, object] = {}

    if dry_run or not api_key:
        if not api_key:
            console.print("[yellow]!!  未设置 AGENTFLOW_API_KEY，使用干跑模式[/yellow]")
        else:
            console.print("[yellow][*]  干跑模式 — 不调用 LLM[/yellow]")

        async def agent_fn(node_id: str, ctx: dict, _stream=None) -> str:
            return f"[dry-run] output from {node_id}"
    else:
        from agentflow.runtime.llm_client import OpenAIClient
        from agentflow.runtime.builder import AgentBuilder
        from agentflow.runtime.memory.manager import MemoryProfile
        from agentflow.runtime.toolkit import tool, ToolKit

        toolkit = ToolKit()
        _loaded_tools: dict[str, object] = {}

        # 从 agentflow.yaml 加载工具
        for ts in config.tools.tools:
            try:
                if ts.kind == "local" and ts.path:
                    module_path, func_name = ts.path.rsplit(":", 1)
                    import importlib
                    mod = importlib.import_module(module_path)
                    func = getattr(mod, func_name)
                    t = tool(name=ts.name, description=ts.description)(func)
                    toolkit.add(t)
                    _loaded_tools[ts.name] = t
                    console.print(f"  [dim]工具 '{ts.name}' 已加载 ({ts.path})[/dim]")
                else:
                    console.print(f"  [yellow]!! 工具 '{ts.name}' (kind={ts.kind}) 暂不支持[/yellow]")
            except Exception as e:
                console.print(f"  [yellow]!! 工具 '{ts.name}' 加载失败: {e}[/yellow]")

        llm = OpenAIClient(
            api_key=api_key, model=model, base_url=base_url,
            max_retries=config.llm.max_retries, timeout=config.llm.timeout,
        )

        _thinking_map = {
            "react": ThinkingMode.REACT,
            "cot": ThinkingMode.COT,
            "plan_execute": ThinkingMode.PLAN_EXECUTE,
            "adaptive": ThinkingMode.ADAPTIVE,
        }
        _memory_map = {
            "light": MemoryProfile.light(),
            "standard": MemoryProfile.standard(),
            "deep": MemoryProfile.deep(),
        }

        for node in wf.nodes:
            if node.kind != NodeKind.AGENT:
                continue
            cfg = node.agent
            builder = AgentBuilder(node.id).with_llm(llm).with_max_iterations(
                config.runtime.max_iterations
            )

            if cfg and cfg.prompt:
                builder.with_prompt(cfg.prompt)

            if cfg and cfg.thinking:
                mode = _thinking_map.get(cfg.thinking, ThinkingMode.ADAPTIVE)
                builder.with_thinking(mode)

            if cfg and cfg.memory:
                profile = _memory_map.get(cfg.memory, MemoryProfile.standard())
                builder.with_memory(profile)

            if cfg and cfg.tools:
                for tool_name in cfg.tools:
                    t = _loaded_tools.get(tool_name)
                    if t:
                        builder.with_tools(t)
                    else:
                        console.print(f"  [yellow]!! 工具 '{tool_name}' 未在 agentflow.yaml 中找到[/yellow]")

            for skill_name in config.discover_skills():
                builder.with_skill(skill_name)

            agent = asyncio.run(builder.build())
            agents[node.id] = agent
            thinking_str = cfg.thinking if cfg else config.runtime.thinking_mode
            memory_str = cfg.memory if cfg else config.runtime.memory_profile
            console.print(f"  [dim]Agent '{node.id}' 已构建 "
                          f"(thinking={thinking_str}, memory={memory_str})[/dim]")

        async def agent_fn(node_id: str, ctx: dict, _stream=None) -> str:
            agent = agents.get(node_id)
            if agent is None:
                return f"[no-agent] node '{node_id}' not registered as AGENT"
            user_input = ctx.get("user_input", "")
            if not user_input:
                prev = ctx.get("previous_outputs", {})
                if prev:
                    user_input = f"Previous outputs: {prev}"
                else:
                    user_input = "Execute your task."
            try:
                result = await agent.run(user_input)
                return result.output
            except Exception as e:
                return f"[error] {e}"

    # 执行
    console.print()
    console.print("[blue]>>  执行 Workflow...[/blue]")
    executor = DAGExecutor()

    results, trace = asyncio.run(executor.execute(wf, agent_fn=agent_fn))

    # 输出结果
    console.print()
    table = Table(title="执行结果", show_header=True, header_style="bold")
    table.add_column("节点", style="cyan")
    table.add_column("状态", style="magenta")
    table.add_column("耗时(ms)", style="dim")
    table.add_column("输出预览", style="green")

    for nid, nr in trace.node_results.items():
        status = "[green]OK[/green]" if nr.success else "[red]X[/red]"
        if nr.skipped_by_condition:
            status = "[dim]skip[/dim]"
        preview = str(nr.output)[:60] if nr.output else "(空)"
        table.add_row(nid, status, str(nr.duration_ms), preview)

    console.print(table)
    console.print(f"[dim]总耗时: {trace.total_duration_ms}ms | "
                  f"分组: {' → '.join(str(g) for g in trace.groups)}[/dim]")

    # 保存 trace
    from agentflow.trace.tracer import TraceStore
    store = TraceStore()
    wid = store.save(trace)
    console.print(f"[dim]Trace 已保存: {wid}[/dim]")


# ---------------------------------------------------------------------------
# eval — 评测
# ---------------------------------------------------------------------------

@app.command()
def eval(
    workflow_path: str = typer.Option("workflow.yaml", help="Workflow YAML 文件路径"),
    suite_name: str = typer.Option("default", help="评测套件名称"),
):
    """运行评测并输出报告。"""
    from agentflow.dsl.serializer import from_yaml
    from agentflow.runtime.orchestrator import DAGExecutor
    from agentflow.eval.suite import EvalSuite, EvalCase
    from agentflow.eval.exact_match import ExactMatchEvaluator
    from agentflow.eval.semantic import SemanticEvaluator

    try:
        wf = from_yaml(workflow_path)
    except FileNotFoundError:
        console.print(f"[red]X 文件不存在: {workflow_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]== 运行评测: '{suite_name}' → '{wf.name}'[/yellow]")

    evaluator = ExactMatchEvaluator()
    cases = [
        EvalCase("c1", "你好", "你好", evaluator),
        EvalCase("c2", "Hello", "Hello", evaluator),
        EvalCase("c3", "Hi", "Hello", evaluator),
    ]
    suite = EvalSuite(suite_name, cases)

    executor = DAGExecutor()

    async def run_workflow(user_input: str) -> str:
        async def agent_fn(node_id: str, ctx: dict) -> str:
            return f"[mock] response to: {user_input} from {node_id}"

        results, _ = await executor.execute(wf, agent_fn=agent_fn)
        last_nid = wf.nodes[-1].id
        return results[last_nid].output if last_nid in results else "(no output)"

    report = asyncio.run(suite.run(run_workflow))

    table = Table(title=f"Eval 报告: {report.name}")
    table.add_column("Case", style="cyan")
    table.add_column("通过", style="magenta")
    table.add_column("分数", style="green")
    table.add_column("原因", style="dim")

    for d in report.details:
        status = "[green]PASS[/green]" if d["passed"] else "[red]FAIL[/red]"
        table.add_row(d["case_id"], status, f"{d['score']:.2f}", d.get("reason", ""))

    console.print(table)
    console.print(f"\n通过率: [bold]{report.pass_rate:.0%}[/bold] "
                  f"({report.passed}/{report.total})")

    low = suite.diagnose(report, min_score=0.5)
    if low:
        console.print()
        console.print("[red]!!  低分案例:[/red]")
        for item in low:
            console.print(f"  [red]•[/red] {item['case_id']}: {item['score']:.2f} — {item['reason']}")


# ---------------------------------------------------------------------------
# trace — 查看轨迹
# ---------------------------------------------------------------------------

@app.command()
def trace(
    trace_id: str = typer.Argument("", help="Trace ID"),
):
    """查看执行轨迹。"""
    if not trace_id:
        console.print("[yellow]用法: agentflow trace <workflow_id>[/yellow]")
        console.print("[dim]运行 agentflow dev 或 agentflow eval 后获取 workflow_id[/dim]")
        return

    from agentflow.trace.tracer import TraceStore
    store = TraceStore()
    t = store.load(trace_id)
    if t is None:
        console.print(f"[red]Trace 未找到: {trace_id}[/red]")
        return

    d = t.to_dict()
    console.print(f"[cyan]Workflow: {d['workflow_name']}[/cyan]")
    console.print(f"[dim]总耗时: {d['summary']['total_duration_ms']}ms | "
                  f"节点: {d['summary']['nodes_executed']} | "
                  f"失败: {d['summary']['nodes_failed']}[/dim]")

    for nid, nt in d.get("node_traces", {}).items():
        console.print(f"\n[bold]{nid}[/bold] — {nt['total_turns']} turns, "
                      f"{nt['total_tool_calls']} tool calls, "
                      f"{nt['total_duration_ms']}ms")
        if nt.get("error"):
            console.print(f"  [red]error: {nt['error']}[/red]")
        for turn in nt.get("turns", []):
            tools_str = ", ".join(
                tc["tool"] for tc in turn.get("tool_calls", [])
            ) or "—"
            console.print(
                f"  turn {turn['turn']}: {turn['finish_reason']} | "
                f"tools=[{tools_str}] | {turn['duration_ms']}ms"
            )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
