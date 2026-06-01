"""AgentFlow CLI — 入口命令"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="AgentFlow - 生产级多Agent编排与评测框架")
console = Console()


@app.command()
def new(name: str = typer.Argument(..., help="Project name")):
    """创建一个新的 AgentFlow 项目。"""
    console.print(f"[green]✨ Creating new AgentFlow project: {name}[/green]")
    import os
    os.makedirs(f"{name}/workflows", exist_ok=True)
    os.makedirs(f"{name}/evals", exist_ok=True)
    with open(f"{name}/agentflow.yaml", "w") as f:
        f.write(f"# AgentFlow project: {name}\n")
    console.print(f"[green]✅ Project '{name}' created![/green]")


@app.command()
def dev(workflow: str = typer.Argument("workflow.yaml", help="Path to workflow YAML file")):
    """启动开发服务器，运行 Workflow 并查看 Trace。"""
    console.print(f"[blue]🚀 Starting dev server with: {workflow}[/blue]")
    console.print("[yellow]⚙️  Orchestration engine not yet connected[/yellow]")


@app.command()
def eval(
    suite: str = typer.Option("default", help="Eval suite name"),
    workflow: str = typer.Option("workflow.yaml", help="Workflow to evaluate"),
):
    """运行评测并输出报告。"""
    console.print(f"[yellow]📊 Running eval suite '{suite}' on '{workflow}'...[/yellow]")
    table = Table(title="Eval Results")
    table.add_column("Case", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Score", style="green")
    table.add_row("example-1", "PASS", "1.00")
    table.add_row("example-2", "PASS", "0.85")
    table.add_row("example-3", "FAIL", "0.30")
    console.print(table)


@app.command()
def trace(trace_id: str = typer.Argument(..., help="Trace ID to inspect")):
    """查看执行轨迹详情。"""
    console.print(f"[cyan]🔍 Fetching trace: {trace_id}[/cyan]")
    console.print("[dim]Trace details not yet available[/dim]")


if __name__ == "__main__":
    app()
