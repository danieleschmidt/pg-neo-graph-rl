#!/usr/bin/env python3
"""
Automation Orchestrator for PG-Neo-Graph-RL.

This script coordinates and manages all automated tasks including:
- Metrics collection and reporting
- Automated testing and quality checks
- Dependency updates and security scanning
- Performance monitoring and alerting
- Deployment automation
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import schedule
import threading
import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AutomationTask:
    """Represents an automation task."""
    name: str
    description: str
    command: str
    schedule: str  # cron-like or interval
    priority: TaskPriority
    timeout: int = 300  # seconds
    retry_count: int = 3
    dependencies: List[str] = None
    environment: Dict[str, str] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_status: TaskStatus = TaskStatus.PENDING
    last_output: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class TaskResult:
    """Result of task execution."""
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: datetime
    execution_time: float
    output: str
    error: Optional[str] = None
    return_code: Optional[int] = None


class AutomationOrchestrator:
    """Main orchestrator for all automation tasks."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file or self.project_root / 'config' / 'automation.json'
        self.tasks: Dict[str, AutomationTask] = {}
        self.task_history: List[TaskResult] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.shutdown_event = asyncio.Event()
        
        # Load configuration
        self._load_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._shutdown())
    
    async def _shutdown(self):
        """Graceful shutdown of orchestrator."""
        self.shutdown_event.set()
        
        # Cancel running tasks
        for task_name, task in self.running_tasks.items():
            self.logger.info(f"Cancelling running task: {task_name}")
            task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        self.logger.info("Shutdown complete")
    
    def _load_config(self):
        """Load automation configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config_data = json.load(f)
                
                # Load task configurations
                for task_data in config_data.get('tasks', []):
                    task = AutomationTask(
                        name=task_data['name'],
                        description=task_data['description'],
                        command=task_data['command'],
                        schedule=task_data['schedule'],
                        priority=TaskPriority(task_data.get('priority', 2)),
                        timeout=task_data.get('timeout', 300),
                        retry_count=task_data.get('retry_count', 3),
                        dependencies=task_data.get('dependencies', []),
                        environment=task_data.get('environment', {}),
                        enabled=task_data.get('enabled', True)
                    )
                    self.tasks[task.name] = task
                
                self.logger.info(f"Loaded {len(self.tasks)} automation tasks")
            else:
                self.logger.warning(f"Config file not found: {self.config_file}")
                self._create_default_config()
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default automation configuration."""
        default_tasks = [
            {
                "name": "daily_metrics_collection",
                "description": "Collect and store daily metrics",
                "command": "python scripts/metrics_collector.py collect",
                "schedule": "daily at 06:00",
                "priority": 2,
                "timeout": 600,
                "enabled": True
            },
            {
                "name": "daily_health_check",
                "description": "Generate daily health report",
                "command": "python scripts/automated_reporting.py run --report daily_health_check",
                "schedule": "daily at 07:00",
                "priority": 2,
                "timeout": 300,
                "dependencies": ["daily_metrics_collection"]
            },
            {
                "name": "code_quality_check",
                "description": "Run comprehensive code quality checks",
                "command": "python -m ruff check . && python -m mypy pg_neo_graph_rl/",
                "schedule": "every 4 hours",
                "priority": 2,
                "timeout": 300
            },
            {
                "name": "security_scan",
                "description": "Run security vulnerability scans",
                "command": "python -m bandit -r pg_neo_graph_rl/ && python -m safety check",
                "schedule": "daily at 02:00",
                "priority": 3,
                "timeout": 600
            },
            {
                "name": "dependency_update_check",
                "description": "Check for dependency updates",
                "command": "pip list --outdated --format=json",
                "schedule": "weekly on monday at 09:00",
                "priority": 2,
                "timeout": 300
            },
            {
                "name": "test_suite_full",
                "description": "Run full test suite with coverage",
                "command": "python -m pytest tests/ --cov=pg_neo_graph_rl --cov-report=html --cov-report=json",
                "schedule": "daily at 01:00",
                "priority": 3,
                "timeout": 1200
            },
            {
                "name": "performance_monitoring",
                "description": "Monitor system performance metrics",
                "command": "python scripts/metrics_collector.py monitor",
                "schedule": "continuous",
                "priority": 1,
                "timeout": 0  # Continuous task
            },
            {
                "name": "weekly_executive_report",
                "description": "Generate weekly executive summary",
                "command": "python scripts/automated_reporting.py run --report weekly_executive_summary",
                "schedule": "weekly on friday at 17:00",
                "priority": 2,
                "timeout": 300
            },
            {
                "name": "database_maintenance",
                "description": "Perform database maintenance tasks",
                "command": "echo 'Database maintenance placeholder'",
                "schedule": "weekly on sunday at 03:00",
                "priority": 1,
                "timeout": 600,
                "enabled": False  # Disabled until database is implemented
            },
            {
                "name": "log_rotation",
                "description": "Rotate and compress old log files",
                "command": "find logs/ -name '*.log' -mtime +7 -exec gzip {} \\;",
                "schedule": "daily at 04:00",
                "priority": 1,
                "timeout": 300
            }
        ]
        
        default_config = {
            "tasks": default_tasks,
            "global_settings": {
                "max_concurrent_tasks": 5,
                "task_timeout_default": 300,
                "retry_delay": 60,
                "log_retention_days": 30,
                "enable_notifications": True,
                "notification_channels": ["email", "slack"]
            }
        }
        
        # Create config directory and file
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.logger.info(f"Created default config at {self.config_file}")
        self._load_config()
    
    async def execute_task(self, task_name: str, force: bool = False) -> TaskResult:
        """Execute a single automation task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task not found: {task_name}")
        
        task = self.tasks[task_name]
        
        if not task.enabled and not force:
            self.logger.info(f"Task {task_name} is disabled, skipping")
            return TaskResult(
                task_name=task_name,
                status=TaskStatus.SKIPPED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                execution_time=0.0,
                output="Task disabled"
            )
        
        # Check dependencies
        if task.dependencies:
            for dep_name in task.dependencies:
                if dep_name in self.running_tasks:
                    self.logger.info(f"Waiting for dependency {dep_name} to complete...")
                    await self.running_tasks[dep_name]
        
        self.logger.info(f"Executing task: {task_name}")
        start_time = datetime.utcnow()
        
        # Prepare environment
        env = os.environ.copy()
        if task.environment:
            env.update(task.environment)
        
        # Execute with retries
        for attempt in range(task.retry_count + 1):
            try:
                # Run the command
                process = await asyncio.create_subprocess_shell(
                    task.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.project_root,
                    env=env
                )
                
                # Wait for completion with timeout
                if task.timeout > 0:
                    stdout, _ = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=task.timeout
                    )
                else:
                    stdout, _ = await process.communicate()
                
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()
                
                output = stdout.decode('utf-8') if stdout else ""
                
                # Update task info
                task.last_run = end_time
                task.execution_time = execution_time
                task.last_output = output
                
                if process.returncode == 0:
                    task.last_status = TaskStatus.COMPLETED
                    self.logger.info(f"Task {task_name} completed successfully in {execution_time:.2f}s")
                    
                    result = TaskResult(
                        task_name=task_name,
                        status=TaskStatus.COMPLETED,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=execution_time,
                        output=output,
                        return_code=process.returncode
                    )
                    
                    self.task_history.append(result)
                    return result
                else:
                    error_msg = f"Task failed with return code {process.returncode}"
                    if attempt < task.retry_count:
                        self.logger.warning(f"{error_msg}, retrying... (attempt {attempt + 1})")
                        await asyncio.sleep(60)  # Wait before retry
                        continue
                    else:
                        task.last_status = TaskStatus.FAILED
                        self.logger.error(f"Task {task_name} failed after {attempt + 1} attempts")
                        
                        result = TaskResult(
                            task_name=task_name,
                            status=TaskStatus.FAILED,
                            start_time=start_time,
                            end_time=end_time,
                            execution_time=execution_time,
                            output=output,
                            error=error_msg,
                            return_code=process.returncode
                        )
                        
                        self.task_history.append(result)
                        return result
            
            except asyncio.TimeoutError:
                error_msg = f"Task timed out after {task.timeout} seconds"
                if attempt < task.retry_count:
                    self.logger.warning(f"{error_msg}, retrying... (attempt {attempt + 1})")
                    continue
                else:
                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    task.last_status = TaskStatus.FAILED
                    task.last_run = end_time
                    
                    result = TaskResult(
                        task_name=task_name,
                        status=TaskStatus.FAILED,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=execution_time,
                        output="",
                        error=error_msg
                    )
                    
                    self.task_history.append(result)
                    return result
            
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                if attempt < task.retry_count:
                    self.logger.warning(f"{error_msg}, retrying... (attempt {attempt + 1})")
                    continue
                else:
                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    task.last_status = TaskStatus.FAILED
                    task.last_run = end_time
                    
                    result = TaskResult(
                        task_name=task_name,
                        status=TaskStatus.FAILED,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=execution_time,
                        output="",
                        error=error_msg
                    )
                    
                    self.task_history.append(result)
                    return result
    
    async def execute_tasks(self, task_names: List[str] = None, force: bool = False) -> List[TaskResult]:
        """Execute multiple tasks."""
        tasks_to_run = task_names or list(self.tasks.keys())
        results = []
        
        # Sort tasks by priority
        sorted_tasks = sorted(
            [(name, self.tasks[name]) for name in tasks_to_run if name in self.tasks],
            key=lambda x: x[1].priority.value,
            reverse=True
        )
        
        # Execute tasks
        for task_name, task in sorted_tasks:
            if self.shutdown_event.is_set():
                break
            
            try:
                # Track running task
                task_coro = self.execute_task(task_name, force)
                self.running_tasks[task_name] = asyncio.create_task(task_coro)
                
                # Execute task
                result = await self.running_tasks[task_name]
                results.append(result)
                
                # Remove from running tasks
                del self.running_tasks[task_name]
            
            except Exception as e:
                self.logger.error(f"Error executing task {task_name}: {e}")
                result = TaskResult(
                    task_name=task_name,
                    status=TaskStatus.FAILED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    execution_time=0.0,
                    output="",
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    async def start_scheduler(self):
        """Start the task scheduler."""
        self.logger.info("Starting automation scheduler...")
        
        # Schedule tasks based on their schedule configuration
        for task_name, task in self.tasks.items():
            if not task.enabled:
                continue
            
            self._schedule_task(task_name, task)
        
        # Main scheduler loop
        while not self.shutdown_event.is_set():
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Clean up completed tasks
                completed_tasks = [
                    name for name, task in self.running_tasks.items()
                    if task.done()
                ]
                for task_name in completed_tasks:
                    del self.running_tasks[task_name]
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
        
        self.logger.info("Scheduler stopped")
    
    def _schedule_task(self, task_name: str, task: AutomationTask):
        """Schedule a single task."""
        schedule_str = task.schedule.lower()
        
        try:
            if schedule_str == "continuous":
                # Don't schedule continuous tasks with the scheduler
                return
            elif "daily at" in schedule_str:
                time_part = schedule_str.split("at")[1].strip()
                schedule.every().day.at(time_part).do(self._run_scheduled_task, task_name)
            elif "weekly on" in schedule_str and "at" in schedule_str:
                parts = schedule_str.split()
                day = parts[2]
                time_part = schedule_str.split("at")[1].strip()
                getattr(schedule.every(), day).at(time_part).do(self._run_scheduled_task, task_name)
            elif "every" in schedule_str and "hours" in schedule_str:
                hours = int(schedule_str.split()[1])
                schedule.every(hours).hours.do(self._run_scheduled_task, task_name)
            elif "every" in schedule_str and "minutes" in schedule_str:
                minutes = int(schedule_str.split()[1])
                schedule.every(minutes).minutes.do(self._run_scheduled_task, task_name)
            else:
                self.logger.warning(f"Unknown schedule format for task {task_name}: {schedule_str}")
        
        except Exception as e:
            self.logger.error(f"Error scheduling task {task_name}: {e}")
    
    def _run_scheduled_task(self, task_name: str):
        """Run a scheduled task (called by scheduler)."""
        if task_name not in self.running_tasks and not self.shutdown_event.is_set():
            asyncio.create_task(self.execute_task(task_name))
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        status = {
            "total_tasks": len(self.tasks),
            "enabled_tasks": len([t for t in self.tasks.values() if t.enabled]),
            "running_tasks": len(self.running_tasks),
            "tasks": {},
            "recent_history": []
        }
        
        # Task details
        for name, task in self.tasks.items():
            status["tasks"][name] = {
                "name": name,
                "description": task.description,
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "last_status": task.last_status.value,
                "execution_time": task.execution_time,
                "priority": task.priority.value,
                "currently_running": name in self.running_tasks
            }
        
        # Recent history (last 10 executions)
        status["recent_history"] = [
            {
                "task_name": result.task_name,
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "execution_time": result.execution_time,
                "error": result.error
            }
            for result in sorted(self.task_history, key=lambda x: x.start_time, reverse=True)[:10]
        ]
        
        return status
    
    def generate_report(self) -> str:
        """Generate automation status report."""
        status = self.get_task_status()
        
        report = f"""# Automation Orchestrator Report

**Generated:** {datetime.utcnow().isoformat()}

## Overview
- Total Tasks: {status['total_tasks']}
- Enabled Tasks: {status['enabled_tasks']}
- Currently Running: {status['running_tasks']}

## Task Status
"""
        
        for task_name, task_info in status["tasks"].items():
            status_emoji = {
                "completed": "✅",
                "failed": "❌", 
                "running": "🏃",
                "pending": "⏳",
                "skipped": "⏭️"
            }.get(task_info["last_status"], "❓")
            
            report += f"- **{task_name}**: {status_emoji} {task_info['last_status']}"
            if task_info["last_run"]:
                report += f" (Last run: {task_info['last_run']})"
            if not task_info["enabled"]:
                report += " [DISABLED]"
            report += "\n"
        
        if status["recent_history"]:
            report += "\n## Recent Execution History\n"
            for execution in status["recent_history"]:
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌", 
                    "running": "🏃",
                    "pending": "⏳",
                    "skipped": "⏭️"
                }.get(execution["status"], "❓")
                
                report += f"- {execution['start_time']}: **{execution['task_name']}** {status_emoji} ({execution['execution_time']:.1f}s)"
                if execution["error"]:
                    report += f" - Error: {execution['error']}"
                report += "\n"
        
        return report


async def main():
    """Main entry point for automation orchestrator."""
    parser = argparse.ArgumentParser(description="PG-Neo-Graph-RL Automation Orchestrator")
    parser.add_argument(
        'command',
        choices=['start', 'run', 'status', 'stop', 'config'],
        help='Command to execute'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='Specific task name to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force execution of disabled tasks'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        config_file = Path(args.config) if args.config else None
        orchestrator = AutomationOrchestrator(config_file)
        
        if args.command == 'start':
            logger.info("Starting automation orchestrator...")
            await orchestrator.start_scheduler()
        
        elif args.command == 'run':
            if args.task:
                logger.info(f"Running task: {args.task}")
                result = await orchestrator.execute_task(args.task, args.force)
                print(f"Task {args.task} completed with status: {result.status.value}")
                if result.error:
                    print(f"Error: {result.error}")
                sys.exit(0 if result.status == TaskStatus.COMPLETED else 1)
            else:
                logger.info("Running all enabled tasks...")
                results = await orchestrator.execute_tasks(force=args.force)
                failed_tasks = [r for r in results if r.status == TaskStatus.FAILED]
                print(f"Executed {len(results)} tasks, {len(failed_tasks)} failed")
                sys.exit(0 if not failed_tasks else 1)
        
        elif args.command == 'status':
            status = orchestrator.get_task_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'config':
            logger.info(f"Configuration file: {orchestrator.config_file}")
            logger.info(f"Tasks configured: {len(orchestrator.tasks)}")
            for task_name, task in orchestrator.tasks.items():
                logger.info(f"  - {task_name}: {task.schedule} ({'enabled' if task.enabled else 'disabled'})")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())