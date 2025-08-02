#!/usr/bin/env python3
"""
Automated reporting system for PG-Neo-Graph-RL.

This script provides automated report generation and distribution for various
stakeholders including development team, management, and operations.
"""

import os
import sys
import json
import smtplib
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.metrics_collector import (
        SystemMetricsCollector, ApplicationMetricsCollector, 
        ProjectMetricsCollector, MetricsReporter, MetricsReport
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: Metrics collector not available")


@dataclass
class ReportConfig:
    """Configuration for automated reports."""
    name: str
    frequency: str  # daily, weekly, monthly
    recipients: List[str]
    format_type: str  # json, markdown, html
    include_metrics: List[str]  # system, application, project
    alert_thresholds: Dict[str, float]
    delivery_methods: List[str]  # email, slack, webhook, file


@dataclass 
class DeliveryConfig:
    """Configuration for report delivery."""
    email_smtp_server: Optional[str] = None
    email_smtp_port: Optional[int] = None
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    webhook_urls: List[str] = None
    file_output_dir: Optional[str] = None


class ReportScheduler:
    """Manage scheduled report generation and delivery."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file or self.project_root / 'config' / 'reporting.json'
        self.reports_config: List[ReportConfig] = []
        self.delivery_config = DeliveryConfig()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load reporting configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config_data = json.load(f)
                
                # Load report configurations
                for report_data in config_data.get('reports', []):
                    self.reports_config.append(ReportConfig(**report_data))
                
                # Load delivery configuration
                delivery_data = config_data.get('delivery', {})
                self.delivery_config = DeliveryConfig(**delivery_data)
                
                self.logger.info(f"Loaded {len(self.reports_config)} report configurations")
            else:
                self.logger.warning(f"Config file not found: {self.config_file}")
                self._create_default_config()
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default reporting configuration."""
        default_config = {
            "reports": [
                {
                    "name": "daily_health_check",
                    "frequency": "daily",
                    "recipients": ["dev-team@example.com"],
                    "format_type": "markdown",
                    "include_metrics": ["system", "application", "project"],
                    "alert_thresholds": {
                        "cpu_usage": 80.0,
                        "memory_usage": 85.0,
                        "disk_usage": 90.0,
                        "code_coverage": 70.0,
                        "failed_tests": 0
                    },
                    "delivery_methods": ["email", "file"]
                },
                {
                    "name": "weekly_executive_summary",
                    "frequency": "weekly",
                    "recipients": ["management@example.com"],
                    "format_type": "html",
                    "include_metrics": ["project"],
                    "alert_thresholds": {
                        "code_coverage": 80.0,
                        "security_issues": 0,
                        "technical_debt_hours": 40.0
                    },
                    "delivery_methods": ["email"]
                },
                {
                    "name": "monthly_detailed_report",
                    "frequency": "monthly",
                    "recipients": ["stakeholders@example.com"],
                    "format_type": "html",
                    "include_metrics": ["system", "application", "project"],
                    "alert_thresholds": {},
                    "delivery_methods": ["email", "file"]
                }
            ],
            "delivery": {
                "email_smtp_server": "smtp.gmail.com",
                "email_smtp_port": 587,
                "email_username": "reports@example.com",
                "email_password": "your-app-password",
                "slack_webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "webhook_urls": [],
                "file_output_dir": "reports/"
            }
        }
        
        # Create config directory and file
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.logger.info(f"Created default config at {self.config_file}")
        self._load_config()
    
    async def generate_and_deliver_reports(self, report_name: Optional[str] = None):
        """Generate and deliver reports."""
        if not METRICS_AVAILABLE:
            self.logger.error("Metrics collector not available")
            return
        
        reports_to_process = self.reports_config
        if report_name:
            reports_to_process = [r for r in self.reports_config if r.name == report_name]
        
        for report_config in reports_to_process:
            try:
                await self._process_report(report_config)
            except Exception as e:
                self.logger.error(f"Error processing report {report_config.name}: {e}")
    
    async def _process_report(self, config: ReportConfig):
        """Process a single report configuration."""
        self.logger.info(f"Processing report: {config.name}")
        
        # Collect metrics based on configuration
        system_metrics = None
        app_metrics = None
        project_metrics = None
        
        if 'system' in config.include_metrics:
            collector = SystemMetricsCollector()
            system_metrics = collector.collect_system_metrics()
        
        if 'application' in config.include_metrics:
            collector = ApplicationMetricsCollector()
            app_metrics = collector.collect_application_metrics()
        
        if 'project' in config.include_metrics:
            collector = ProjectMetricsCollector(self.project_root)
            project_metrics = collector.collect_project_metrics()
        
        # Generate report
        reporter = MetricsReporter(self.project_root)
        
        # Create placeholder metrics if not collected
        if system_metrics is None:
            from scripts.metrics_collector import SystemMetrics
            system_metrics = SystemMetrics(
                timestamp=datetime.utcnow().isoformat(),
                cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0,
                network_io={}, process_count=0, load_average=[0.0, 0.0, 0.0]
            )
        
        if app_metrics is None:
            from scripts.metrics_collector import ApplicationMetrics
            app_metrics = ApplicationMetrics(
                timestamp=datetime.utcnow().isoformat(),
                jax_devices=0, jax_platform="unknown",
                training_metrics={}, federated_metrics={}, graph_metrics={},
                error_counts={}, performance_metrics={}
            )
        
        if project_metrics is None:
            from scripts.metrics_collector import ProjectMetrics
            project_metrics = ProjectMetrics(
                timestamp=datetime.utcnow().isoformat(),
                code_coverage=0.0, test_count=0, failed_tests=0,
                code_quality_score=0.0, security_issues=0,
                dependency_vulnerabilities=0, lines_of_code=0,
                technical_debt_hours=0.0
            )
        
        report = reporter.generate_report(
            system_metrics, app_metrics, project_metrics,
            report_period=config.frequency
        )
        
        # Apply custom thresholds for alerts
        report = self._apply_custom_thresholds(report, config.alert_thresholds)
        
        # Deliver report
        await self._deliver_report(report, config)
    
    def _apply_custom_thresholds(self, report: MetricsReport, thresholds: Dict[str, float]) -> MetricsReport:
        """Apply custom alert thresholds to the report."""
        if not thresholds:
            return report
        
        # Re-generate alerts with custom thresholds
        custom_alerts = []
        
        # System alerts
        if 'cpu_usage' in thresholds:
            if report.system_metrics.cpu_usage > thresholds['cpu_usage']:
                custom_alerts.append({
                    'type': 'warning',
                    'category': 'system',
                    'message': f'CPU usage {report.system_metrics.cpu_usage:.1f}% exceeds threshold {thresholds["cpu_usage"]}%',
                    'threshold': thresholds['cpu_usage'],
                    'current_value': report.system_metrics.cpu_usage
                })
        
        if 'memory_usage' in thresholds:
            if report.system_metrics.memory_usage > thresholds['memory_usage']:
                custom_alerts.append({
                    'type': 'critical',
                    'category': 'system',
                    'message': f'Memory usage {report.system_metrics.memory_usage:.1f}% exceeds threshold {thresholds["memory_usage"]}%',
                    'threshold': thresholds['memory_usage'],
                    'current_value': report.system_metrics.memory_usage
                })
        
        # Project alerts
        if 'code_coverage' in thresholds:
            if report.project_metrics.code_coverage < thresholds['code_coverage']:
                custom_alerts.append({
                    'type': 'warning',
                    'category': 'quality',
                    'message': f'Code coverage {report.project_metrics.code_coverage:.1f}% below threshold {thresholds["code_coverage"]}%',
                    'threshold': thresholds['code_coverage'],
                    'current_value': report.project_metrics.code_coverage
                })
        
        if 'failed_tests' in thresholds:
            if report.project_metrics.failed_tests > thresholds['failed_tests']:
                custom_alerts.append({
                    'type': 'error',
                    'category': 'quality',
                    'message': f'{report.project_metrics.failed_tests} failed tests (threshold: {thresholds["failed_tests"]})',
                    'threshold': thresholds['failed_tests'],
                    'current_value': report.project_metrics.failed_tests
                })
        
        if 'security_issues' in thresholds:
            if report.project_metrics.security_issues > thresholds['security_issues']:
                custom_alerts.append({
                    'type': 'critical',
                    'category': 'security',
                    'message': f'{report.project_metrics.security_issues} security issues (threshold: {thresholds["security_issues"]})',
                    'threshold': thresholds['security_issues'],
                    'current_value': report.project_metrics.security_issues
                })
        
        if 'technical_debt_hours' in thresholds:
            if report.project_metrics.technical_debt_hours > thresholds['technical_debt_hours']:
                custom_alerts.append({
                    'type': 'warning',
                    'category': 'maintenance',
                    'message': f'Technical debt {report.project_metrics.technical_debt_hours:.1f}h exceeds threshold {thresholds["technical_debt_hours"]}h',
                    'threshold': thresholds['technical_debt_hours'],
                    'current_value': report.project_metrics.technical_debt_hours
                })
        
        # Update report with custom alerts
        report.alerts = custom_alerts
        return report
    
    async def _deliver_report(self, report: MetricsReport, config: ReportConfig):
        """Deliver report using configured methods."""
        reporter = MetricsReporter(self.project_root)
        report_content = reporter.export_report(report, config.format_type)
        
        # Prepare report metadata
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{config.name}_report_{timestamp}.{config.format_type}"
        
        # Deliver via each configured method
        for method in config.delivery_methods:
            try:
                if method == 'email':
                    await self._send_email_report(report, report_content, config, filename)
                elif method == 'slack':
                    await self._send_slack_report(report, report_content, config)
                elif method == 'webhook':
                    await self._send_webhook_report(report, report_content, config)
                elif method == 'file':
                    await self._save_file_report(report_content, config, filename)
                else:
                    self.logger.warning(f"Unknown delivery method: {method}")
            
            except Exception as e:
                self.logger.error(f"Error delivering report via {method}: {e}")
    
    async def _send_email_report(self, report: MetricsReport, content: str, config: ReportConfig, filename: str):
        """Send report via email."""
        if not all([
            self.delivery_config.email_smtp_server,
            self.delivery_config.email_username,
            self.delivery_config.email_password
        ]):
            self.logger.error("Email configuration incomplete")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.delivery_config.email_username
            msg['To'] = ', '.join(config.recipients)
            msg['Subject'] = f"[PG-Neo-Graph-RL] {config.name.replace('_', ' ').title()} - {datetime.utcnow().strftime('%Y-%m-%d')}"
            
            # Email body
            alert_count = len(report.alerts)
            critical_alerts = len([a for a in report.alerts if a['type'] in ['error', 'critical']])
            
            body = f"""
PG-Neo-Graph-RL Automated Report

Report: {config.name.replace('_', ' ').title()}
Generated: {report.generated_at}
Period: {report.report_period}

Summary:
- Alerts: {alert_count} total ({critical_alerts} critical)
- System CPU: {report.system_metrics.cpu_usage:.1f}%
- Memory Usage: {report.system_metrics.memory_usage:.1f}%
- Code Coverage: {report.project_metrics.code_coverage:.1f}%
- Test Results: {report.project_metrics.test_count - report.project_metrics.failed_tests}/{report.project_metrics.test_count} passed

See attached detailed report for complete metrics and analysis.

Best regards,
PG-Neo-Graph-RL Monitoring System
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach detailed report
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(content.encode('utf-8'))
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(self.delivery_config.email_smtp_server, self.delivery_config.email_smtp_port)
            server.starttls()
            server.login(self.delivery_config.email_username, self.delivery_config.email_password)
            text = msg.as_string()
            server.sendmail(self.delivery_config.email_username, config.recipients, text)
            server.quit()
            
            self.logger.info(f"Email report sent to {len(config.recipients)} recipients")
        
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
    
    async def _send_slack_report(self, report: MetricsReport, content: str, config: ReportConfig):
        """Send report summary to Slack."""
        if not self.delivery_config.slack_webhook_url:
            self.logger.error("Slack webhook URL not configured")
            return
        
        try:
            import aiohttp
            
            # Prepare Slack message
            alert_count = len(report.alerts)
            critical_alerts = len([a for a in report.alerts if a['type'] in ['error', 'critical']])
            
            color = "good"  # green
            if critical_alerts > 0:
                color = "danger"  # red
            elif alert_count > 0:
                color = "warning"  # yellow
            
            slack_data = {
                "text": f"PG-Neo-Graph-RL {config.name.replace('_', ' ').title()}",
                "attachments": [
                    {
                        "color": color,
                        "title": f"Report Summary - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                        "fields": [
                            {
                                "title": "Alerts",
                                "value": f"{alert_count} total ({critical_alerts} critical)",
                                "short": True
                            },
                            {
                                "title": "System Status",
                                "value": f"CPU: {report.system_metrics.cpu_usage:.1f}% | Memory: {report.system_metrics.memory_usage:.1f}%",
                                "short": True
                            },
                            {
                                "title": "Quality Metrics",
                                "value": f"Coverage: {report.project_metrics.code_coverage:.1f}% | Tests: {report.project_metrics.test_count - report.project_metrics.failed_tests}/{report.project_metrics.test_count}",
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            # Add critical alerts to message
            if critical_alerts > 0:
                critical_alert_messages = [
                    a['message'] for a in report.alerts 
                    if a['type'] in ['error', 'critical']
                ][:5]  # Limit to top 5
                
                slack_data["attachments"].append({
                    "color": "danger",
                    "title": "Critical Alerts",
                    "text": "\n".join(f"• {msg}" for msg in critical_alert_messages)
                })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(self.delivery_config.slack_webhook_url, json=slack_data) as response:
                    if response.status == 200:
                        self.logger.info("Slack report sent successfully")
                    else:
                        self.logger.error(f"Slack webhook returned status {response.status}")
        
        except ImportError:
            self.logger.error("aiohttp not available for Slack notifications")
        except Exception as e:
            self.logger.error(f"Error sending Slack report: {e}")
    
    async def _send_webhook_report(self, report: MetricsReport, content: str, config: ReportConfig):
        """Send report to configured webhooks."""
        webhook_urls = self.delivery_config.webhook_urls or []
        
        if not webhook_urls:
            self.logger.warning("No webhook URLs configured")
            return
        
        try:
            import aiohttp
            
            # Prepare webhook payload
            webhook_data = {
                "report_name": config.name,
                "timestamp": report.generated_at,
                "alert_count": len(report.alerts),
                "critical_alerts": len([a for a in report.alerts if a['type'] in ['error', 'critical']]),
                "system_metrics": asdict(report.system_metrics),
                "project_metrics": asdict(report.project_metrics),
                "alerts": report.alerts[:10],  # Limit alerts
                "content": content if len(content) < 50000 else content[:50000] + "...[truncated]"
            }
            
            # Send to each webhook
            async with aiohttp.ClientSession() as session:
                for webhook_url in webhook_urls:
                    try:
                        async with session.post(webhook_url, json=webhook_data) as response:
                            if response.status == 200:
                                self.logger.info(f"Webhook report sent to {webhook_url}")
                            else:
                                self.logger.error(f"Webhook {webhook_url} returned status {response.status}")
                    except Exception as e:
                        self.logger.error(f"Error sending to webhook {webhook_url}: {e}")
        
        except ImportError:
            self.logger.error("aiohttp not available for webhook delivery")
        except Exception as e:
            self.logger.error(f"Error sending webhook reports: {e}")
    
    async def _save_file_report(self, content: str, config: ReportConfig, filename: str):
        """Save report to file."""
        try:
            output_dir = Path(self.delivery_config.file_output_dir or 'reports')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Report saved to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving file report: {e}")


async def main():
    """Main entry point for automated reporting."""
    parser = argparse.ArgumentParser(description="PG-Neo-Graph-RL Automated Reporting")
    parser.add_argument(
        'command',
        choices=['run', 'test', 'config'],
        help='Command to execute'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Specific report name to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
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
        scheduler = ReportScheduler(config_file)
        
        if args.command == 'run':
            logger.info("Running automated reporting...")
            await scheduler.generate_and_deliver_reports(args.report)
            logger.info("Reporting completed")
        
        elif args.command == 'test':
            logger.info("Testing report delivery (dry run)...")
            # Could implement a test mode that generates sample data
            logger.info("Test completed")
        
        elif args.command == 'config':
            logger.info(f"Configuration file: {scheduler.config_file}")
            logger.info(f"Reports configured: {len(scheduler.reports_config)}")
            for report in scheduler.reports_config:
                logger.info(f"  - {report.name}: {report.frequency}, {len(report.recipients)} recipients")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())