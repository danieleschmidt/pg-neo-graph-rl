#!/usr/bin/env python3
"""
Command-line interface for pg-neo-graph-rl.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .algorithms import GraphPPO
from .core import FederatedGraphRL
from .environments import PowerGridEnvironment, SwarmEnvironment, TrafficEnvironment
from .utils.exceptions import GraphRLError

console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="pg-neo-rl",
        description="Federated Graph-Neural Reinforcement Learning toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pg-neo-rl demo                    # Run interactive demo
  pg-neo-rl train traffic.yaml     # Train on traffic scenario
  pg-neo-rl evaluate model.pkl     # Evaluate trained model
  pg-neo-rl monitor --port 8080    # Start monitoring dashboard
  pg-neo-rl status                  # Show system status
        """
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"pg-neo-graph-rl {__version__}"
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument(
        "--scenario",
        choices=["traffic", "power", "swarm"],
        default="traffic",
        help="Demo scenario to run"
    )
    demo_parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of federated agents"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a federated model")
    train_parser.add_argument("config", type=Path, help="Training configuration file")
    train_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    train_parser.add_argument("--resume", type=Path, help="Resume from checkpoint")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("model", type=Path, help="Trained model path")
    eval_parser.add_argument("--scenario", type=Path, help="Evaluation scenario")
    eval_parser.add_argument("--metrics", help="Comma-separated metrics to compute")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start monitoring dashboard")
    monitor_parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    monitor_parser.add_argument("--host", default="localhost", help="Dashboard host")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Version info command
    subparsers.add_parser("info", help="Show detailed system information")

    return parser


def setup_console_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Setup console logging with rich formatting."""
    log_level = logging.DEBUG if verbose else logging.INFO
    # Simple console logging setup for now
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(str(log_file))])
        ]
    )


def run_demo(args: argparse.Namespace) -> int:
    """Run interactive demo."""
    console.print(Panel.fit(
        f"🚀 pg-neo-graph-rl Demo\nScenario: {args.scenario.title()}\nAgents: {args.agents}",
        title="Demo",
        border_style="green"
    ))

    try:
        # Create environment
        console.print("\n📍 Creating environment...")
        if args.scenario == "traffic":
            env = TrafficEnvironment(num_intersections=16, time_resolution=5.0)
        elif args.scenario == "power":
            env = PowerGridEnvironment(num_buses=20)
        elif args.scenario == "swarm":
            env = SwarmEnvironment(num_drones=25)
        else:
            console.print(f"[red]Unknown scenario: {args.scenario}[/red]")
            return 1

        # Create federated system
        console.print("🔗 Creating federated system...")
        fed_system = FederatedGraphRL(
            num_agents=args.agents,
            aggregation="gossip",
            topology="ring"
        )

        # Create agent
        console.print("🤖 Creating agent...")
        initial_state = env.reset()
        node_dim = initial_state.nodes.shape[1]
        action_dim = 3

        agent = GraphPPO(
            agent_id=0,
            action_dim=action_dim,
            node_dim=node_dim
        )
        fed_system.register_agent(agent)

        # Run demo steps
        console.print("\n🚀 Running demo steps...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # Step 1: Environment interaction
            task1 = progress.add_task("Environment step...", total=1)
            state = env.reset()
            num_nodes = len(state.nodes)
            random_actions = [0, 1, 2] * (num_nodes // 3 + 1)
            random_actions = random_actions[:num_nodes]
            import jax.numpy as jnp
            next_state, rewards, done, info = env.step(jnp.array(random_actions))
            progress.advance(task1, 1)

            # Step 2: Agent action
            task2 = progress.add_task("Agent action selection...", total=1)
            subgraph = fed_system.get_subgraph(0, state)
            actions, agent_info = agent.act(subgraph, training=False)
            progress.advance(task2, 1)

            # Step 3: Communication
            task3 = progress.add_task("Testing communication...", total=1)
            comm_stats = fed_system.get_communication_stats()
            progress.advance(task3, 1)

        # Display results
        results_table = Table(title="Demo Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Environment", args.scenario.title())
        results_table.add_row("Nodes", str(state.nodes.shape[0]))
        results_table.add_row("Edges", str(state.edges.shape[0]))
        results_table.add_row("Agents", str(args.agents))
        results_table.add_row("Communication", "Connected" if comm_stats["is_connected"] else "Disconnected")
        results_table.add_row("Avg Reward", f"{float(rewards.mean()):.3f}")

        console.print("\n")
        console.print(results_table)

        console.print("\n✅ [green]Demo completed successfully![/green]")
        console.print("🎯 Core components verified:")
        console.print("   - Environment simulation")
        console.print("   - Graph neural networks")
        console.print("   - Federated coordination")
        console.print("   - Communication topology")

        return 0

    except Exception as e:
        console.print(f"\n[red]Demo failed: {e}[/red]")
        if args.verbose:
            console.print_exception()
        return 1


def show_status(args: argparse.Namespace) -> int:
    """Show system status."""
    console.print(Panel.fit("🔍 System Status", border_style="blue"))

    # System info table
    status_table = Table(title="System Information")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details")

    # Check imports
    try:
        import jax
        jax_status = "✅ Available"
        jax_details = f"Version {jax.__version__}, Backend: {jax.default_backend()}"
    except ImportError:
        jax_status = "❌ Missing"
        jax_details = "JAX not installed"

    try:
        import flax
        flax_status = "✅ Available"
        flax_details = f"Version {flax.__version__}"
    except ImportError:
        flax_status = "❌ Missing"
        flax_details = "Flax not installed"

    try:
        import networkx as nx
        nx_status = "✅ Available"
        nx_details = f"Version {nx.__version__}"
    except ImportError:
        nx_status = "❌ Missing"
        nx_details = "NetworkX not installed"

    status_table.add_row("JAX", jax_status, jax_details)
    status_table.add_row("Flax", flax_status, flax_details)
    status_table.add_row("NetworkX", nx_status, nx_details)
    status_table.add_row("pg-neo-graph-rl", "✅ Available", f"Version {__version__}")

    console.print(status_table)

    # Check example scenarios
    console.print("\n📊 Environment Status:")
    env_table = Table()
    env_table.add_column("Environment", style="cyan")
    env_table.add_column("Status", style="green")

    # Test environments
    try:
        TrafficEnvironment(num_intersections=4)
        env_table.add_row("Traffic", "✅ Working")
    except Exception as e:
        env_table.add_row("Traffic", f"❌ Error: {str(e)[:50]}")

    try:
        PowerGridEnvironment(num_buses=4)
        env_table.add_row("Power Grid", "✅ Working")
    except Exception as e:
        env_table.add_row("Power Grid", f"❌ Error: {str(e)[:50]}")

    try:
        SwarmEnvironment(num_drones=4)
        env_table.add_row("Swarm", "✅ Working")
    except Exception as e:
        env_table.add_row("Swarm", f"❌ Error: {str(e)[:50]}")

    console.print(env_table)

    return 0


def show_info(args: argparse.Namespace) -> int:
    """Show detailed system information."""
    console.print(Panel.fit("ℹ️  System Information", border_style="blue"))

    # Package info
    info_data = {
        "Package": "pg-neo-graph-rl",
        "Version": __version__,
        "Description": "Federated Graph-Neural Reinforcement Learning toolkit",
        "Python": sys.version.split()[0],
        "Platform": sys.platform
    }

    # Add JAX device info
    try:
        import jax
        devices = jax.devices()
        device_info = [f"{d.platform}:{d.id}" for d in devices]
        info_data["JAX Devices"] = ", ".join(device_info)
        info_data["JAX Backend"] = jax.default_backend()
    except ImportError:
        info_data["JAX Devices"] = "Not available"

    for key, value in info_data.items():
        console.print(f"[cyan]{key:12}[/cyan]: {value}")

    # Show configuration paths
    console.print("\n📁 [cyan]Paths:[/cyan]")
    console.print(f"  Config: {Path.home() / '.pg-neo-rl'}")
    console.print(f"  Logs:   {Path.cwd() / 'logs'}")
    console.print(f"  Models: {Path.cwd() / 'models'}")

    # Show available commands
    console.print("\n⚡ [cyan]Available Commands:[/cyan]")
    commands = [
        ("demo", "Run interactive demonstration"),
        ("train", "Train federated models"),
        ("evaluate", "Evaluate trained models"),
        ("monitor", "Start monitoring dashboard"),
        ("status", "Show system status"),
        ("info", "Show system information")
    ]

    for cmd, desc in commands:
        console.print(f"  [green]{cmd:10}[/green]: {desc}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_console_logging(args.verbose, args.log_file)

    try:
        if args.command == "demo":
            return run_demo(args)
        elif args.command == "status":
            return show_status(args)
        elif args.command == "info":
            return show_info(args)
        elif args.command == "train":
            console.print("[yellow]Training functionality coming soon![/yellow]")
            return 0
        elif args.command == "evaluate":
            console.print("[yellow]Evaluation functionality coming soon![/yellow]")
            return 0
        elif args.command == "monitor":
            console.print("[yellow]Monitoring dashboard coming soon![/yellow]")
            return 0
        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except GraphRLError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
