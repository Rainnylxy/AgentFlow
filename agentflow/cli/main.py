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
    llm_model: str = typer.Option("gpt-4o", help="使用的 LLM 模型"),
    dry_run: bool = typer.Option(False, help="干跑模式——不调 LLM，只看 DAG 执行"),
):
    """启动开发服务器，运行 Workflow 并查看 Trace。"""
    from agentflow.dsl.serializer import from_yaml
    from agentflow.runtime.orchestrator import DAGExecutor

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

    # 创建 agent_fn
    if dry_run:

        async def agent_fn(node_id: str, ctx: dict) -> str:
            return f"[dry-run] output from {node_id}"

        console.print("[yellow][*]  干跑模式 — 不调用 LLM[/yellow]")
    else:
        # 尝试连接 LLM
        api_key = os.getenv("AGENTFLOW_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            console.print("[yellow]!!  未设置 AGENTFLOW_API_KEY，使用干跑模式[/yellow]")

            async def agent_fn(node_id: str, ctx: dict) -> str:
                return f"[no-llm] output from {node_id}"
        else:
            from agentflow.runtime.llm_client import OpenAIClient

            llm = OpenAIClient(
                api_key=api_key,
                model=llm_model,
            )

            async def agent_fn(node_id: str, ctx: dict) -> str:
                msgs = ctx.get("incoming_messages", [])
                prev = ctx.get("previous_outputs", {})
                prompt = f"Task for node '{node_id}'.\n"
                if prev:
                    prompt += f"Previous outputs: {list(prev.keys())}\n"
                if msgs:
                    prompt += f"Incoming messages: {[m.to_dict() for m in msgs]}\n"
                try:
                    resp = await llm.chat([{"role": "user", "content": prompt}])
                    return resp.content
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

    # 加载 Workflow
    try:
        wf = from_yaml(workflow_path)
    except FileNotFoundError:
        console.print(f"[red]X 文件不存在: {workflow_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]== 运行评测: '{suite_name}' → '{wf.name}'[/yellow]")

    # 构建评测套件
    evaluator = ExactMatchEvaluator()
    cases = [
        EvalCase("c1", "你好", "你好", evaluator),
        EvalCase("c2", "Hello", "Hello", evaluator),
        EvalCase("c3", "Hi", "Hello", evaluator),
    ]
    suite = EvalSuite(suite_name, cases)

    executor = DAGExecutor()

    async def run_workflow(user_input: str) -> str:
        """包装 Workflow 执行为 EvalSuite 需要的 agent_fn。"""
        # 把 user_input 注入到第一个节点的输入
        async def agent_fn(node_id: str, ctx: dict) -> str:
            return f"[mock] response to: {user_input} from {node_id}"

        results, _ = await executor.execute(wf, agent_fn=agent_fn)
        # 返回最终节点的输出
        last_nid = wf.nodes[-1].id
        return results[last_nid].output if last_nid in results else "(no output)"

    report = asyncio.run(suite.run(run_workflow))

    # 输出报告
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

    # 诊断低分 case
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
    """查看最近的执行轨迹。"""
    from agentflow.trace.client import TraceClient

    client = TraceClient()

    if not trace_id:
        console.print("[yellow]tip: 运行 'agentflow dev' 后这里会显示 Trace ID[/yellow]")
        console.print("[dim]用法: agentflow trace <trace_id>[/dim]")
        return

    console.print(f"[cyan]>> Trace: {trace_id}[/cyan]")

    # 从内存 TraceClient 查找
    traces = client._traces if hasattr(client, '_traces') else {}
    t = traces.get(trace_id)
    if not t:
        console.print(f"[red]X Trace 未找到: {trace_id}[/red]")
        return

    # 显示
    table = Table(title=f"Trace: {t.trace_id}")
    table.add_column("Span", style="cyan")
    table.add_column("状态", style="magenta")
    table.add_column("耗时(ms)", style="dim")
    table.add_column("输出", style="green")

    for span in t.spans:
        status = "[green]✓[/green]" if span.status == "success" else "[red]✗[/red]"
        table.add_row(span.name, status, str(span.duration_ms), span.output[:60])

    console.print(table)
    console.print(f"[dim]Workflow: {t.workflow_id} | 状态: {t.status}[/dim]")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
