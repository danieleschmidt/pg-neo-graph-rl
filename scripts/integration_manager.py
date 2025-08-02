#!/usr/bin/env python3
"""
Integration Manager for PG-Neo-Graph-RL Automation Stack.

This script manages the integration and coordination of all automation
components including metrics collection, reporting, orchestration, and
monitoring dashboard.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.metrics_collector import main as metrics_main
    from scripts.automated_reporting import main as reporting_main
    from scripts.automation_orchestrator import AutomationOrchestrator
    from scripts.monitoring_dashboard import MonitoringDashboard
    SCRIPTS_AVAILABLE = True
except ImportError:
    SCRIPTS_AVAILABLE = False
    print("Warning: Some automation scripts not available")


@dataclass
class ServiceConfig:
    """Configuration for a managed service."""
    name: str
    command: str
    args: List[str]
    enabled: bool = True
    restart_on_failure: bool = True
    max_restarts: int = 5
    restart_delay: int = 10
    health_check_interval: int = 30
    environment: Optional[Dict[str, str]] = None


class ServiceManager:
    """Manage automation services."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services: Dict[str, ServiceConfig] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.restart_counts: Dict[str, int] = {}
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    def register_service(self, config: ServiceConfig):
        """Register a service for management."""
        self.services[config.name] = config
        self.restart_counts[config.name] = 0
        self.logger.info(f"Registered service: {config.name}")
    
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            self.logger.error(f"Service not found: {service_name}")
            return False
        
        config = self.services[service_name]
        
        if not config.enabled:
            self.logger.info(f"Service {service_name} is disabled")
            return False
        
        if service_name in self.processes:
            self.logger.warning(f"Service {service_name} is already running")
            return True
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if config.environment:
                env.update(config.environment)
            
            # Start the process
            cmd = [config.command] + config.args
            self.logger.info(f"Starting {service_name}: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=Path(__file__).parent.parent
            )
            
            self.processes[service_name] = process
            self.logger.info(f"Service {service_name} started with PID {process.pid}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start {service_name}: {e}")
            return False
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.processes:
            self.logger.warning(f"Service {service_name} is not running")
            return True
        
        try:
            process = self.processes[service_name]
            self.logger.info(f"Stopping {service_name} (PID {process.pid})")
            
            # Send SIGTERM
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Service {service_name} did not stop gracefully, killing...")
                process.kill()
                process.wait()
            
            del self.processes[service_name]
            self.logger.info(f"Service {service_name} stopped")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping {service_name}: {e}")
            return False
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        await self.stop_service(service_name)
        await asyncio.sleep(2)  # Brief pause
        return await self.start_service(service_name)
    
    async def start_all_services(self):
        """Start all enabled services."""
        for service_name, config in self.services.items():
            if config.enabled:
                await self.start_service(service_name)
                await asyncio.sleep(2)  # Stagger startup
    
    async def stop_all_services(self):
        """Stop all running services."""
        for service_name in list(self.processes.keys()):
            await self.stop_service(service_name)
    
    async def health_check_loop(self):
        """Monitor service health and restart if needed."""
        while not self.shutdown_event.is_set():
            try:
                for service_name, config in self.services.items():
                    if not config.enabled:
                        continue
                    
                    # Check if process is running
                    if service_name in self.processes:
                        process = self.processes[service_name]
                        
                        # Check if process has exited
                        return_code = process.poll()
                        if return_code is not None:
                            self.logger.warning(f"Service {service_name} exited with code {return_code}")
                            del self.processes[service_name]
                            
                            # Restart if configured to do so
                            if config.restart_on_failure:
                                restart_count = self.restart_counts.get(service_name, 0)
                                if restart_count < config.max_restarts:
                                    self.restart_counts[service_name] = restart_count + 1
                                    self.logger.info(f"Restarting {service_name} (attempt {restart_count + 1})")
                                    await asyncio.sleep(config.restart_delay)
                                    await self.start_service(service_name)
                                else:
                                    self.logger.error(f"Service {service_name} has exceeded max restarts ({config.max_restarts})")
                    
                    elif config.enabled and config.restart_on_failure:
                        # Service should be running but isn't
                        self.logger.info(f"Service {service_name} should be running, starting...")
                        await self.start_service(service_name)
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Gracefully shutdown all services."""
        self.logger.info("Shutting down service manager...")
        self.shutdown_event.set()
        await self.stop_all_services()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            'services': {},
            'summary': {
                'total': len(self.services),
                'running': len(self.processes),
                'enabled': len([s for s in self.services.values() if s.enabled])
            }
        }
        
        for service_name, config in self.services.items():
            is_running = service_name in self.processes
            pid = self.processes[service_name].pid if is_running else None
            
            status['services'][service_name] = {
                'name': service_name,
                'enabled': config.enabled,
                'running': is_running,
                'pid': pid,
                'restart_count': self.restart_counts.get(service_name, 0),
                'max_restarts': config.max_restarts,
                'restart_on_failure': config.restart_on_failure
            }
        
        return status


class IntegrationManager:
    """Main integration manager for all automation components."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file or self.project_root / 'config' / 'integration.json'
        
        # Service manager
        self.service_manager = ServiceManager()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load integration configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config_data = json.load(f)
                
                # Load service configurations
                for service_data in config_data.get('services', []):
                    service_config = ServiceConfig(
                        name=service_data['name'],
                        command=service_data['command'],
                        args=service_data.get('args', []),
                        enabled=service_data.get('enabled', True),
                        restart_on_failure=service_data.get('restart_on_failure', True),
                        max_restarts=service_data.get('max_restarts', 5),
                        restart_delay=service_data.get('restart_delay', 10),
                        health_check_interval=service_data.get('health_check_interval', 30),
                        environment=service_data.get('environment')
                    )
                    self.service_manager.register_service(service_config)
                
                self.logger.info(f"Loaded {len(config_data.get('services', []))} service configurations")
            else:
                self.logger.warning(f"Config file not found: {self.config_file}")
                self._create_default_config()
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default integration configuration."""
        default_services = [
            {
                "name": "metrics_collector",
                "command": "python",
                "args": ["scripts/metrics_collector.py", "monitor"],
                "enabled": True,
                "restart_on_failure": True,
                "max_restarts": 10,
                "restart_delay": 30,
                "environment": {
                    "PYTHONPATH": "."
                }
            },
            {
                "name": "automation_orchestrator",
                "command": "python",
                "args": ["scripts/automation_orchestrator.py", "start"],
                "enabled": True,
                "restart_on_failure": True,
                "max_restarts": 5,
                "restart_delay": 60,
                "environment": {
                    "PYTHONPATH": "."
                }
            },
            {
                "name": "monitoring_dashboard",
                "command": "python",
                "args": ["scripts/monitoring_dashboard.py", "--host", "0.0.0.0", "--port", "8080"],
                "enabled": True,
                "restart_on_failure": True,
                "max_restarts": 5,
                "restart_delay": 30,
                "environment": {
                    "PYTHONPATH": "."
                }
            },
            {
                "name": "daily_reporting",
                "command": "python",
                "args": ["scripts/automated_reporting.py", "run"],
                "enabled": False,  # Run via orchestrator instead
                "restart_on_failure": False,
                "max_restarts": 0
            }
        ]
        
        default_config = {
            "services": default_services,
            "global_settings": {
                "log_level": "INFO",
                "health_check_interval": 30,
                "startup_delay": 2,
                "shutdown_timeout": 30
            }
        }
        
        # Create config directory and file
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.logger.info(f"Created default config at {self.config_file}")
        self._load_config()
    
    async def start_integration(self):
        """Start the complete integration stack."""
        self.logger.info("Starting PG-Neo-Graph-RL automation integration...")
        
        # Ensure required directories exist
        directories = ['logs', 'metrics', 'reports', 'config']
        for directory in directories:
            (self.project_root / directory).mkdir(exist_ok=True)
        
        # Start all services
        await self.service_manager.start_all_services()
        
        # Start health monitoring
        health_check_task = asyncio.create_task(self.service_manager.health_check_loop())
        
        self.logger.info("Integration stack started successfully")
        self.logger.info("Access the monitoring dashboard at: http://localhost:8080")
        
        try:
            # Keep running until shutdown
            await self.service_manager.shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            # Clean shutdown
            health_check_task.cancel()
            await self.service_manager.shutdown()
    
    async def stop_integration(self):
        """Stop the integration stack."""
        self.logger.info("Stopping integration stack...")
        await self.service_manager.shutdown()
    
    async def restart_integration(self):
        """Restart the integration stack."""
        await self.stop_integration()
        await asyncio.sleep(5)
        await self.start_integration()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the integration."""
        service_status = self.service_manager.get_status()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'integration_status': 'running' if service_status['summary']['running'] > 0 else 'stopped',
            'services': service_status,
            'endpoints': {
                'monitoring_dashboard': 'http://localhost:8080',
                'api_metrics': 'http://localhost:8080/api/metrics',
                'api_health': 'http://localhost:8080/api/health'
            },
            'directories': {
                'config': str(self.project_root / 'config'),
                'logs': str(self.project_root / 'logs'),
                'metrics': str(self.project_root / 'metrics'),
                'reports': str(self.project_root / 'reports')
            }
        }
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Check services
        service_status = self.service_manager.get_status()
        health['checks']['services'] = {
            'status': 'healthy' if service_status['summary']['running'] > 0 else 'unhealthy',
            'details': service_status
        }
        
        # Check directories
        directories = ['logs', 'metrics', 'reports', 'config']
        dir_status = 'healthy'
        dir_details = {}
        
        for directory in directories:
            dir_path = self.project_root / directory
            exists = dir_path.exists()
            writable = os.access(dir_path, os.W_OK) if exists else False
            
            dir_details[directory] = {
                'exists': exists,
                'writable': writable,
                'path': str(dir_path)
            }
            
            if not exists or not writable:
                dir_status = 'unhealthy'
        
        health['checks']['directories'] = {
            'status': dir_status,
            'details': dir_details
        }
        
        # Check Python dependencies
        required_packages = ['aiohttp', 'psutil', 'schedule']
        dep_status = 'healthy'
        dep_details = {}
        
        for package in required_packages:
            try:
                __import__(package)
                dep_details[package] = 'available'
            except ImportError:
                dep_details[package] = 'missing'
                dep_status = 'degraded'
        
        health['checks']['dependencies'] = {
            'status': dep_status,
            'details': dep_details
        }
        
        # Overall status
        check_statuses = [check['status'] for check in health['checks'].values()]
        if 'unhealthy' in check_statuses:
            health['overall_status'] = 'unhealthy'
        elif 'degraded' in check_statuses:
            health['overall_status'] = 'degraded'
        
        return health


async def main():
    """Main entry point for integration manager."""
    parser = argparse.ArgumentParser(description="PG-Neo-Graph-RL Integration Manager")
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'restart', 'status', 'health', 'config'],
        help='Command to execute'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--service',
        type=str,
        help='Specific service name (for start/stop/restart)'
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
        manager = IntegrationManager(config_file)
        
        if args.command == 'start':
            if args.service:
                success = await manager.service_manager.start_service(args.service)
                sys.exit(0 if success else 1)
            else:
                await manager.start_integration()
        
        elif args.command == 'stop':
            if args.service:
                success = await manager.service_manager.stop_service(args.service)
                sys.exit(0 if success else 1)
            else:
                await manager.stop_integration()
        
        elif args.command == 'restart':
            if args.service:
                success = await manager.service_manager.restart_service(args.service)
                sys.exit(0 if success else 1)
            else:
                await manager.restart_integration()
        
        elif args.command == 'status':
            status = manager.get_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'health':
            health = manager.run_health_check()
            print(json.dumps(health, indent=2, default=str))
            
            # Exit with error code if unhealthy
            if health['overall_status'] == 'unhealthy':
                sys.exit(1)
        
        elif args.command == 'config':
            logger.info(f"Configuration file: {manager.config_file}")
            logger.info(f"Services configured: {len(manager.service_manager.services)}")
            
            for service_name, config in manager.service_manager.services.items():
                logger.info(f"  - {service_name}: {config.command} {' '.join(config.args)} ({'enabled' if config.enabled else 'disabled'})")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())