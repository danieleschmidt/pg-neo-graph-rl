#!/usr/bin/env python3
"""
Training CLI for pg-neo-graph-rl.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..utils.exceptions import TrainingError
from .config import TrainingConfig
from .trainer import FederatedTrainer

console = Console()


def create_default_config(output_path: Path) -> None:
    """Create a default training configuration file."""
    config = TrainingConfig()
    config.to_yaml(output_path)
    console.print(f"[green]Default configuration created: {output_path}[/green]")


def train_command(config_path: Path, output_dir: Optional[Path] = None, resume_path: Optional[Path] = None) -> int:
    """Execute training command."""
    try:
        # Load configuration
        config = TrainingConfig.from_yaml(config_path)

        # Override output directory if specified
        if output_dir:
            config.output_dir = output_dir
            config.checkpoint_dir = output_dir / "checkpoints"
            config.log_dir = output_dir / "logs"

        console.print(Panel.fit(
            f"🚀 Training Configuration\n"
            f"Environment: {config.environment_type}\n"
            f"Agents: {config.num_agents}\n"
            f"Algorithm: {config.algorithm.upper()}\n"
            f"Episodes: {config.total_episodes}",
            title="Federated Training",
            border_style="green"
        ))

        # Initialize trainer
        console.print("🔧 Initializing trainer...")
        trainer = FederatedTrainer(config)

        # Resume from checkpoint if specified
        if resume_path and resume_path.exists():
            console.print(f"📂 Loading checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)

        # Start training
        console.print("🚀 Starting federated training...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:

            # Add training progress task
            training_task = progress.add_task(
                "Training...",
                total=config.total_episodes,
                completed=trainer.episode
            )

            # Custom training loop with progress updates
            original_train = trainer._train_episode

            def train_episode_with_progress():
                result = original_train()
                progress.advance(training_task, 1)

                # Update description with latest metrics
                if trainer.episode % config.log_frequency == 0:
                    progress.update(
                        training_task,
                        description=f"Episode {trainer.episode}: reward={result['avg_reward']:.3f}"
                    )

                return result

            trainer._train_episode = train_episode_with_progress

            # Execute training
            final_metrics = trainer.train()

        # Training completed
        console.print("\n✅ [green]Training completed successfully![/green]")
        console.print(f"🏆 Best reward: {final_metrics['best_reward']:.3f}")
        console.print(f"📊 Total episodes: {final_metrics['total_episodes']}")
        console.print(f"⏱️  Training time: {final_metrics['training_time_seconds']:.1f}s")
        console.print(f"💾 Results saved to: {config.output_dir}")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        return 130
    except TrainingError as e:
        console.print(f"[red]Training error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print_exception()
        return 1


def main() -> int:
    """Training CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pgnrl-train",
        description="Train federated graph reinforcement learning models"
    )

    subparsers = parser.add_subparsers(dest="command", help="Training commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("config", type=Path, help="Training configuration file")
    train_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    train_parser.add_argument("--resume", type=Path, help="Resume from checkpoint")

    # Config command
    config_parser = subparsers.add_parser("config", help="Create default configuration")
    config_parser.add_argument("output", type=Path, help="Output configuration file")

    args = parser.parse_args()

    if args.command == "train":
        return train_command(args.config, args.output, args.resume)
    elif args.command == "config":
        create_default_config(args.output)
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
