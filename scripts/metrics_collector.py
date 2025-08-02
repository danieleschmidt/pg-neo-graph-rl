#!/usr/bin/env python3
"""
Comprehensive metrics collection and reporting automation for PG-Neo-Graph-RL.

This script provides automated metrics collection, analysis, and reporting
for the federated graph reinforcement learning system.
"""

import os
import sys
import json
import argparse
import logging
import asyncio
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pg_neo_graph_rl.monitoring import (
        get_metrics_collector, initialize_metrics,
        get_health_manager, initialize_health_checks
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring modules not available")


@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: str
    jax_devices: int
    jax_platform: str
    training_metrics: Dict[str, float]
    federated_metrics: Dict[str, float]
    graph_metrics: Dict[str, int]
    error_counts: Dict[str, int]
    performance_metrics: Dict[str, float]


@dataclass
class ProjectMetrics:
    """Project development metrics."""
    timestamp: str
    code_coverage: float
    test_count: int
    failed_tests: int
    code_quality_score: float
    security_issues: int
    dependency_vulnerabilities: int
    lines_of_code: int
    technical_debt_hours: float


@dataclass
class MetricsReport:
    """Comprehensive metrics report."""
    generated_at: str
    report_period: str
    system_metrics: SystemMetrics
    application_metrics: ApplicationMetrics
    project_metrics: ProjectMetrics
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


class SystemMetricsCollector:
    """Collect system-level metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # CPU and memory
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.used / disk.total * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process info
            process_count = len(psutil.pids())
            load_average = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                timestamp=datetime.utcnow().isoformat(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average
            )
            
        except ImportError:
            self.logger.warning("psutil not available, using basic metrics")
            return SystemMetrics(
                timestamp=datetime.utcnow().isoformat(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            raise


class ApplicationMetricsCollector:
    """Collect application-specific metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect current application metrics."""
        metrics = ApplicationMetrics(
            timestamp=datetime.utcnow().isoformat(),
            jax_devices=0,
            jax_platform="unknown",
            training_metrics={},
            federated_metrics={},
            graph_metrics={},
            error_counts={},
            performance_metrics={}
        )
        
        # JAX metrics
        try:
            import jax
            devices = jax.devices()
            metrics.jax_devices = len(devices)
            metrics.jax_platform = devices[0].platform if devices else "unknown"
        except ImportError:
            self.logger.warning("JAX not available for metrics collection")
        
        # Application metrics from monitoring system
        if MONITORING_AVAILABLE:
            try:
                collector = get_metrics_collector()
                if collector:
                    # Get training metrics
                    training_loss = getattr(collector, '_training_loss_values', [])
                    if training_loss:
                        metrics.training_metrics['latest_loss'] = training_loss[-1]
                        metrics.training_metrics['avg_loss'] = sum(training_loss) / len(training_loss)
                    
                    # Get federated learning metrics
                    comm_latencies = getattr(collector, '_communication_latencies', [])
                    if comm_latencies:
                        metrics.federated_metrics['avg_comm_latency'] = sum(comm_latencies) / len(comm_latencies)
                        metrics.federated_metrics['max_comm_latency'] = max(comm_latencies)
                    
                    # Get performance metrics
                    jax_compile_times = getattr(collector, '_jax_compile_times', [])
                    if jax_compile_times:
                        metrics.performance_metrics['avg_compile_time'] = sum(jax_compile_times) / len(jax_compile_times)
                        
            except Exception as e:
                self.logger.warning(f"Error collecting application metrics: {e}")
        
        return metrics


class ProjectMetricsCollector:
    """Collect project development and quality metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def collect_project_metrics(self) -> ProjectMetrics:
        """Collect project development metrics."""
        metrics = ProjectMetrics(
            timestamp=datetime.utcnow().isoformat(),
            code_coverage=0.0,
            test_count=0,
            failed_tests=0,
            code_quality_score=0.0,
            security_issues=0,
            dependency_vulnerabilities=0,
            lines_of_code=0,
            technical_debt_hours=0.0
        )
        
        # Code coverage
        metrics.code_coverage = self._get_code_coverage()
        
        # Test metrics
        test_results = self._run_tests()
        metrics.test_count = test_results.get('total', 0)
        metrics.failed_tests = test_results.get('failed', 0)
        
        # Lines of code
        metrics.lines_of_code = self._count_lines_of_code()
        
        # Code quality
        metrics.code_quality_score = self._assess_code_quality()
        
        # Security and vulnerabilities
        security_results = self._check_security()
        metrics.security_issues = security_results.get('issues', 0)
        metrics.dependency_vulnerabilities = security_results.get('vulnerabilities', 0)
        
        # Technical debt estimation
        metrics.technical_debt_hours = self._estimate_technical_debt()
        
        return metrics
    
    def _get_code_coverage(self) -> float:
        """Get code coverage percentage."""
        try:
            # Run coverage analysis
            result = subprocess.run([
                'python', '-m', 'pytest', '--cov=pg_neo_graph_rl', 
                '--cov-report=json', '--cov-report=term-missing',
                'tests/'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=300)
            
            # Parse coverage report
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    return coverage_data.get('totals', {}).get('percent_covered', 0.0)
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error getting code coverage: {e}")
        
        return 0.0
    
    def _run_tests(self) -> Dict[str, int]:
        """Run tests and get results."""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', '--json-report', '--json-report-file=test-report.json',
                'tests/'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=600)
            
            # Parse test report
            report_file = self.project_root / 'test-report.json'
            if report_file.exists():
                with open(report_file) as f:
                    test_data = json.load(f)
                    summary = test_data.get('summary', {})
                    return {
                        'total': summary.get('total', 0),
                        'passed': summary.get('passed', 0),
                        'failed': summary.get('failed', 0),
                        'skipped': summary.get('skipped', 0),
                        'error': summary.get('error', 0)
                    }
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error running tests: {e}")
        
        return {'total': 0, 'failed': 0}
    
    def _count_lines_of_code(self) -> int:
        """Count lines of code in the project."""
        try:
            total_lines = 0
            python_files = list(self.project_root.glob('**/*.py'))
            
            for file_path in python_files:
                if 'venv' in str(file_path) or '.tox' in str(file_path):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
                        total_lines += lines
                except Exception:
                    continue
            
            return total_lines
        
        except Exception as e:
            self.logger.warning(f"Error counting lines of code: {e}")
            return 0
    
    def _assess_code_quality(self) -> float:
        """Assess code quality using multiple metrics."""
        try:
            # Run ruff for linting
            result = subprocess.run([
                'ruff', 'check', '--output-format=json', 'pg_neo_graph_rl/'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            issues = 0
            if result.stdout:
                try:
                    lint_data = json.loads(result.stdout)
                    issues = len(lint_data) if isinstance(lint_data, list) else 0
                except json.JSONDecodeError:
                    issues = result.stdout.count('\n')
            
            # Calculate quality score (0-100)
            # Fewer issues = higher score
            lines_of_code = max(self._count_lines_of_code(), 1)
            issue_ratio = issues / lines_of_code
            quality_score = max(0, 100 - (issue_ratio * 1000))
            
            return min(100.0, quality_score)
        
        except Exception as e:
            self.logger.warning(f"Error assessing code quality: {e}")
            return 75.0  # Default reasonable score
    
    def _check_security(self) -> Dict[str, int]:
        """Check for security issues and vulnerabilities."""
        results = {'issues': 0, 'vulnerabilities': 0}
        
        try:
            # Run bandit for security issues
            bandit_result = subprocess.run([
                'bandit', '-r', 'pg_neo_graph_rl/', '-f', 'json'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if bandit_result.stdout:
                try:
                    bandit_data = json.loads(bandit_result.stdout)
                    results['issues'] = len(bandit_data.get('results', []))
                except json.JSONDecodeError:
                    pass
            
            # Run safety for dependency vulnerabilities
            safety_result = subprocess.run([
                'safety', 'check', '--json'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if safety_result.stdout:
                try:
                    safety_data = json.loads(safety_result.stdout)
                    results['vulnerabilities'] = len(safety_data) if isinstance(safety_data, list) else 0
                except json.JSONDecodeError:
                    pass
        
        except Exception as e:
            self.logger.warning(f"Error checking security: {e}")
        
        return results
    
    def _estimate_technical_debt(self) -> float:
        """Estimate technical debt in hours."""
        try:
            # This is a simplified estimation based on:
            # - Number of TODO/FIXME comments
            # - Code complexity
            # - Test coverage gaps
            
            todo_count = 0
            complexity_indicators = 0
            
            python_files = list(self.project_root.glob('**/*.py'))
            
            for file_path in python_files:
                if 'venv' in str(file_path) or '.tox' in str(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Count TODOs and FIXMEs
                        todo_count += content.lower().count('todo')
                        todo_count += content.lower().count('fixme')
                        todo_count += content.lower().count('hack')
                        
                        # Count complexity indicators
                        complexity_indicators += content.count('if ')
                        complexity_indicators += content.count('for ')
                        complexity_indicators += content.count('while ')
                        complexity_indicators += content.count('except')
                
                except Exception:
                    continue
            
            # Simple estimation: 
            # - 0.5 hours per TODO/FIXME
            # - Coverage gap penalty
            # - Complexity factor
            
            coverage = self._get_code_coverage()
            coverage_gap_penalty = max(0, (80 - coverage) * 0.1)  # Penalty for coverage < 80%
            
            estimated_hours = (
                todo_count * 0.5 +
                complexity_indicators * 0.01 +
                coverage_gap_penalty
            )
            
            return round(estimated_hours, 2)
        
        except Exception as e:
            self.logger.warning(f"Error estimating technical debt: {e}")
            return 0.0


class MetricsReporter:
    """Generate comprehensive metrics reports."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self, 
        system_metrics: SystemMetrics,
        app_metrics: ApplicationMetrics,
        project_metrics: ProjectMetrics,
        report_period: str = "current"
    ) -> MetricsReport:
        """Generate comprehensive metrics report."""
        
        # Generate alerts based on metrics
        alerts = self._generate_alerts(system_metrics, app_metrics, project_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(system_metrics, app_metrics, project_metrics)
        
        return MetricsReport(
            generated_at=datetime.utcnow().isoformat(),
            report_period=report_period,
            system_metrics=system_metrics,
            application_metrics=app_metrics,
            project_metrics=project_metrics,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _generate_alerts(
        self, 
        system: SystemMetrics, 
        app: ApplicationMetrics, 
        project: ProjectMetrics
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on metric thresholds."""
        alerts = []
        
        # System alerts
        if system.cpu_usage > 80:
            alerts.append({
                'type': 'warning',
                'category': 'system',
                'message': f'High CPU usage: {system.cpu_usage:.1f}%',
                'threshold': 80,
                'current_value': system.cpu_usage
            })
        
        if system.memory_usage > 85:
            alerts.append({
                'type': 'critical',
                'category': 'system',
                'message': f'High memory usage: {system.memory_usage:.1f}%',
                'threshold': 85,
                'current_value': system.memory_usage
            })
        
        if system.disk_usage > 90:
            alerts.append({
                'type': 'critical',
                'category': 'system',
                'message': f'High disk usage: {system.disk_usage:.1f}%',
                'threshold': 90,
                'current_value': system.disk_usage
            })
        
        # Application alerts
        if app.jax_devices == 0:
            alerts.append({
                'type': 'error',
                'category': 'application',
                'message': 'No JAX devices available',
                'threshold': 1,
                'current_value': 0
            })
        
        # Project alerts
        if project.code_coverage < 70:
            alerts.append({
                'type': 'warning',
                'category': 'quality',
                'message': f'Low code coverage: {project.code_coverage:.1f}%',
                'threshold': 70,
                'current_value': project.code_coverage
            })
        
        if project.failed_tests > 0:
            alerts.append({
                'type': 'error',
                'category': 'quality',
                'message': f'{project.failed_tests} test(s) failing',
                'threshold': 0,
                'current_value': project.failed_tests
            })
        
        if project.security_issues > 0:
            alerts.append({
                'type': 'critical',
                'category': 'security',
                'message': f'{project.security_issues} security issue(s) found',
                'threshold': 0,
                'current_value': project.security_issues
            })
        
        if project.dependency_vulnerabilities > 0:
            alerts.append({
                'type': 'critical',
                'category': 'security',
                'message': f'{project.dependency_vulnerabilities} dependency vulnerability(ies)',
                'threshold': 0,
                'current_value': project.dependency_vulnerabilities
            })
        
        return alerts
    
    def _generate_recommendations(
        self, 
        system: SystemMetrics, 
        app: ApplicationMetrics, 
        project: ProjectMetrics
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # System recommendations
        if system.memory_usage > 75:
            recommendations.append(
                "Consider optimizing memory usage or increasing available memory"
            )
        
        if system.cpu_usage > 70:
            recommendations.append(
                "High CPU usage detected - consider profiling for performance bottlenecks"
            )
        
        # Application recommendations
        if app.jax_devices < 2 and app.jax_platform == "cpu":
            recommendations.append(
                "Consider using GPU acceleration for better performance"
            )
        
        # Project recommendations
        if project.code_coverage < 80:
            recommendations.append(
                f"Increase test coverage from {project.code_coverage:.1f}% to at least 80%"
            )
        
        if project.code_quality_score < 85:
            recommendations.append(
                "Address code quality issues identified by linting tools"
            )
        
        if project.technical_debt_hours > 40:
            recommendations.append(
                f"Consider allocating time to address {project.technical_debt_hours:.1f} hours of technical debt"
            )
        
        if not recommendations:
            recommendations.append("All metrics are within acceptable ranges - good job!")
        
        return recommendations
    
    def export_report(self, report: MetricsReport, format_type: str = "json") -> str:
        """Export report in specified format."""
        if format_type.lower() == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        
        elif format_type.lower() == "markdown":
            return self._format_markdown_report(report)
        
        elif format_type.lower() == "html":
            return self._format_html_report(report)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _format_markdown_report(self, report: MetricsReport) -> str:
        """Format report as Markdown."""
        md = f"""# Metrics Report

**Generated:** {report.generated_at}  
**Period:** {report.report_period}

## System Metrics

- **CPU Usage:** {report.system_metrics.cpu_usage:.1f}%
- **Memory Usage:** {report.system_metrics.memory_usage:.1f}%
- **Disk Usage:** {report.system_metrics.disk_usage:.1f}%
- **Process Count:** {report.system_metrics.process_count}
- **Load Average:** {', '.join(f'{x:.2f}' for x in report.system_metrics.load_average)}

## Application Metrics

- **JAX Devices:** {report.application_metrics.jax_devices} ({report.application_metrics.jax_platform})
- **Training Metrics:** {len(report.application_metrics.training_metrics)} collected
- **Federated Metrics:** {len(report.application_metrics.federated_metrics)} collected
- **Performance Metrics:** {len(report.application_metrics.performance_metrics)} collected

## Project Metrics

- **Code Coverage:** {report.project_metrics.code_coverage:.1f}%
- **Total Tests:** {report.project_metrics.test_count}
- **Failed Tests:** {report.project_metrics.failed_tests}
- **Code Quality Score:** {report.project_metrics.code_quality_score:.1f}/100
- **Lines of Code:** {report.project_metrics.lines_of_code:,}
- **Security Issues:** {report.project_metrics.security_issues}
- **Dependency Vulnerabilities:** {report.project_metrics.dependency_vulnerabilities}
- **Technical Debt:** {report.project_metrics.technical_debt_hours:.1f} hours

## Alerts

"""
        if report.alerts:
            for alert in report.alerts:
                emoji = {"error": "🔴", "critical": "⚠️", "warning": "🟡"}.get(alert['type'], "ℹ️")
                md += f"- {emoji} **{alert['category'].title()}:** {alert['message']}\n"
        else:
            md += "✅ No alerts\n"
        
        md += "\n## Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- {rec}\n"
        
        return md
    
    def _format_html_report(self, report: MetricsReport) -> str:
        """Format report as HTML."""
        # Basic HTML template - could be enhanced with CSS/charts
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Metrics Report - {report.generated_at}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ margin: 10px 0; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .error {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .critical {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .warning {{ background-color: #fffde7; border-left: 4px solid #ffeb3b; }}
    </style>
</head>
<body>
    <h1>Metrics Report</h1>
    <p><strong>Generated:</strong> {report.generated_at}</p>
    <p><strong>Period:</strong> {report.report_period}</p>
    
    <h2>System Metrics</h2>
    <div class="metric">CPU Usage: {report.system_metrics.cpu_usage:.1f}%</div>
    <div class="metric">Memory Usage: {report.system_metrics.memory_usage:.1f}%</div>
    <div class="metric">Disk Usage: {report.system_metrics.disk_usage:.1f}%</div>
    
    <h2>Project Metrics</h2>
    <div class="metric">Code Coverage: {report.project_metrics.code_coverage:.1f}%</div>
    <div class="metric">Code Quality: {report.project_metrics.code_quality_score:.1f}/100</div>
    <div class="metric">Technical Debt: {report.project_metrics.technical_debt_hours:.1f} hours</div>
    
    <h2>Alerts</h2>"""
        
        if report.alerts:
            for alert in report.alerts:
                html += f'<div class="alert {alert["type"]}">{alert["message"]}</div>'
        else:
            html += '<div class="metric">✅ No alerts</div>'
        
        html += """
    <h2>Recommendations</h2>
    <ul>"""
        
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
    </ul>
</body>
</html>"""
        
        return html


async def main():
    """Main entry point for metrics collection."""
    parser = argparse.ArgumentParser(description="PG-Neo-Graph-RL Metrics Collector")
    parser.add_argument(
        'command',
        choices=['collect', 'report', 'monitor', 'alert'],
        help='Command to execute'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown', 'html'],
        default='json',
        help='Output format for reports'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='current',
        help='Reporting period'
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
    project_root = Path(__file__).parent.parent
    
    try:
        if args.command == 'collect':
            logger.info("Collecting metrics...")
            
            # Initialize collectors
            system_collector = SystemMetricsCollector()
            app_collector = ApplicationMetricsCollector()
            project_collector = ProjectMetricsCollector(project_root)
            
            # Collect metrics
            system_metrics = system_collector.collect_system_metrics()
            app_metrics = app_collector.collect_application_metrics()
            project_metrics = project_collector.collect_project_metrics()
            
            # Store metrics
            metrics_dir = project_root / 'metrics'
            metrics_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            with open(metrics_dir / f'system_metrics_{timestamp}.json', 'w') as f:
                json.dump(asdict(system_metrics), f, indent=2, default=str)
            
            with open(metrics_dir / f'app_metrics_{timestamp}.json', 'w') as f:
                json.dump(asdict(app_metrics), f, indent=2, default=str)
            
            with open(metrics_dir / f'project_metrics_{timestamp}.json', 'w') as f:
                json.dump(asdict(project_metrics), f, indent=2, default=str)
            
            logger.info(f"Metrics collected and stored in {metrics_dir}")
        
        elif args.command == 'report':
            logger.info("Generating metrics report...")
            
            # Collect current metrics
            system_collector = SystemMetricsCollector()
            app_collector = ApplicationMetricsCollector()
            project_collector = ProjectMetricsCollector(project_root)
            
            system_metrics = system_collector.collect_system_metrics()
            app_metrics = app_collector.collect_application_metrics()
            project_metrics = project_collector.collect_project_metrics()
            
            # Generate report
            reporter = MetricsReporter(project_root)
            report = reporter.generate_report(
                system_metrics, app_metrics, project_metrics, args.period
            )
            
            # Export report
            report_content = reporter.export_report(report, args.format)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {args.output}")
            else:
                print(report_content)
        
        elif args.command == 'monitor':
            logger.info("Starting continuous monitoring...")
            
            # Continuous monitoring loop
            system_collector = SystemMetricsCollector()
            
            while True:
                try:
                    system_metrics = system_collector.collect_system_metrics()
                    
                    # Check for alerts
                    if system_metrics.cpu_usage > 90:
                        logger.warning(f"High CPU usage: {system_metrics.cpu_usage:.1f}%")
                    
                    if system_metrics.memory_usage > 90:
                        logger.warning(f"High memory usage: {system_metrics.memory_usage:.1f}%")
                    
                    if system_metrics.disk_usage > 95:
                        logger.critical(f"Critical disk usage: {system_metrics.disk_usage:.1f}%")
                    
                    # Wait before next check
                    await asyncio.sleep(60)  # Check every minute
                
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)
        
        elif args.command == 'alert':
            logger.info("Checking for alerts...")
            
            # Collect metrics and check for alerts
            system_collector = SystemMetricsCollector()
            app_collector = ApplicationMetricsCollector()
            project_collector = ProjectMetricsCollector(project_root)
            
            system_metrics = system_collector.collect_system_metrics()
            app_metrics = app_collector.collect_application_metrics()
            project_metrics = project_collector.collect_project_metrics()
            
            # Generate alerts
            reporter = MetricsReporter(project_root)
            alerts = reporter._generate_alerts(system_metrics, app_metrics, project_metrics)
            
            if alerts:
                logger.warning(f"Found {len(alerts)} alert(s):")
                for alert in alerts:
                    level = getattr(logger, alert['type'], logger.info)
                    level(f"{alert['category']}: {alert['message']}")
                
                # Exit with error code if critical alerts
                critical_alerts = [a for a in alerts if a['type'] in ['error', 'critical']]
                if critical_alerts:
                    sys.exit(1)
            else:
                logger.info("No alerts found - all metrics within acceptable ranges")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())