#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for PG-Neo-Graph-RL.

This script provides a web-based dashboard for monitoring system metrics,
automation tasks, application health, and project status in real-time.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from aiohttp import web, WSMsgType
    import aiohttp_cors
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available, web dashboard disabled")

try:
    from scripts.metrics_collector import (
        SystemMetricsCollector, ApplicationMetricsCollector, 
        ProjectMetricsCollector, MetricsReporter
    )
    from scripts.automation_orchestrator import AutomationOrchestrator
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring modules not available")


class MonitoringDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent
        
        # Data collectors
        self.system_collector = SystemMetricsCollector() if MONITORING_AVAILABLE else None
        self.app_collector = ApplicationMetricsCollector() if MONITORING_AVAILABLE else None
        self.project_collector = ProjectMetricsCollector(self.project_root) if MONITORING_AVAILABLE else None
        self.orchestrator = AutomationOrchestrator() if MONITORING_AVAILABLE else None
        
        # WebSocket connections
        self.websockets = set()
        
        # Background tasks
        self.background_tasks = []
        
        # Cached data
        self.cached_data = {
            'system_metrics': {},
            'app_metrics': {},
            'project_metrics': {},
            'automation_status': {},
            'alerts': [],
            'last_update': None
        }
    
    async def create_app(self):
        """Create the web application."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        
        app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Routes
        app.router.add_get('/', self.index)
        app.router.add_get('/api/metrics', self.api_metrics)
        app.router.add_get('/api/automation', self.api_automation)
        app.router.add_get('/api/health', self.api_health)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_static('/static', self.project_root / 'dashboard' / 'static', name='static')
        
        # Add CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        return app
    
    async def index(self, request):
        """Serve the main dashboard page."""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    async def api_metrics(self, request):
        """API endpoint for current metrics."""
        try:
            data = await self._collect_all_metrics()
            return web.json_response(data)
        except Exception as e:
            self.logger.error(f"Error in metrics API: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_automation(self, request):
        """API endpoint for automation status."""
        try:
            if self.orchestrator:
                status = self.orchestrator.get_task_status()
                return web.json_response(status)
            else:
                return web.json_response({'error': 'Automation orchestrator not available'}, status=503)
        except Exception as e:
            self.logger.error(f"Error in automation API: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_health(self, request):
        """API endpoint for health check."""
        try:
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'dashboard_uptime': self._get_uptime(),
                'components': {
                    'metrics_collector': MONITORING_AVAILABLE,
                    'automation_orchestrator': MONITORING_AVAILABLE,
                    'websocket_connections': len(self.websockets)
                }
            }
            return web.json_response(health_data)
        except Exception as e:
            self.logger.error(f"Error in health API: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def websocket_handler(self, request):
        """WebSocket handler for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.websockets)}")
        
        try:
            # Send initial data
            await self._send_ws_update(ws)
            
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
        
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        
        finally:
            self.websockets.discard(ws)
            self.logger.info(f"WebSocket disconnected. Total connections: {len(self.websockets)}")
        
        return ws
    
    async def _handle_ws_message(self, ws, data):
        """Handle incoming WebSocket messages."""
        message_type = data.get('type')
        
        if message_type == 'get_metrics':
            metrics = await self._collect_all_metrics()
            await ws.send_str(json.dumps({
                'type': 'metrics_update',
                'data': metrics
            }))
        
        elif message_type == 'get_automation':
            if self.orchestrator:
                status = self.orchestrator.get_task_status()
                await ws.send_str(json.dumps({
                    'type': 'automation_update',
                    'data': status
                }))
        
        elif message_type == 'run_task':
            task_name = data.get('task_name')
            if self.orchestrator and task_name:
                try:
                    result = await self.orchestrator.execute_task(task_name, force=True)
                    await ws.send_str(json.dumps({
                        'type': 'task_result',
                        'task_name': task_name,
                        'result': {
                            'status': result.status.value,
                            'execution_time': result.execution_time,
                            'error': result.error
                        }
                    }))
                except Exception as e:
                    await ws.send_str(json.dumps({
                        'type': 'task_error',
                        'task_name': task_name,
                        'error': str(e)
                    }))
    
    async def _collect_all_metrics(self):
        """Collect all available metrics."""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': {},
            'app_metrics': {},
            'project_metrics': {},
            'alerts': []
        }
        
        try:
            if self.system_collector:
                system_metrics = self.system_collector.collect_system_metrics()
                data['system_metrics'] = {
                    'cpu_usage': system_metrics.cpu_usage,
                    'memory_usage': system_metrics.memory_usage,
                    'disk_usage': system_metrics.disk_usage,
                    'process_count': system_metrics.process_count,
                    'load_average': system_metrics.load_average,
                    'network_io': system_metrics.network_io
                }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        try:
            if self.app_collector:
                app_metrics = self.app_collector.collect_application_metrics()
                data['app_metrics'] = {
                    'jax_devices': app_metrics.jax_devices,
                    'jax_platform': app_metrics.jax_platform,
                    'training_metrics': app_metrics.training_metrics,
                    'federated_metrics': app_metrics.federated_metrics,
                    'performance_metrics': app_metrics.performance_metrics
                }
        except Exception as e:
            self.logger.error(f"Error collecting app metrics: {e}")
        
        try:
            if self.project_collector:
                project_metrics = self.project_collector.collect_project_metrics()
                data['project_metrics'] = {
                    'code_coverage': project_metrics.code_coverage,
                    'test_count': project_metrics.test_count,
                    'failed_tests': project_metrics.failed_tests,
                    'code_quality_score': project_metrics.code_quality_score,
                    'security_issues': project_metrics.security_issues,
                    'dependency_vulnerabilities': project_metrics.dependency_vulnerabilities,
                    'lines_of_code': project_metrics.lines_of_code,
                    'technical_debt_hours': project_metrics.technical_debt_hours
                }
        except Exception as e:
            self.logger.error(f"Error collecting project metrics: {e}")
        
        # Generate alerts
        data['alerts'] = self._generate_alerts(data)
        
        # Cache the data
        self.cached_data = data
        
        return data
    
    def _generate_alerts(self, data):
        """Generate alerts based on current metrics."""
        alerts = []
        
        # System alerts
        system = data.get('system_metrics', {})
        if system.get('cpu_usage', 0) > 80:
            alerts.append({
                'type': 'warning',
                'category': 'system',
                'message': f"High CPU usage: {system['cpu_usage']:.1f}%",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        if system.get('memory_usage', 0) > 85:
            alerts.append({
                'type': 'critical',
                'category': 'system',
                'message': f"High memory usage: {system['memory_usage']:.1f}%",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        if system.get('disk_usage', 0) > 90:
            alerts.append({
                'type': 'critical',
                'category': 'system',
                'message': f"High disk usage: {system['disk_usage']:.1f}%",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Application alerts
        app = data.get('app_metrics', {})
        if app.get('jax_devices', 0) == 0:
            alerts.append({
                'type': 'error',
                'category': 'application',
                'message': "No JAX devices available",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Project alerts
        project = data.get('project_metrics', {})
        if project.get('failed_tests', 0) > 0:
            alerts.append({
                'type': 'error',
                'category': 'quality',
                'message': f"{project['failed_tests']} test(s) failing",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        if project.get('security_issues', 0) > 0:
            alerts.append({
                'type': 'critical',
                'category': 'security',
                'message': f"{project['security_issues']} security issue(s) found",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return alerts
    
    async def _send_ws_update(self, ws):
        """Send update to a specific WebSocket."""
        try:
            data = await self._collect_all_metrics()
            await ws.send_str(json.dumps({
                'type': 'full_update',
                'data': data
            }))
        except Exception as e:
            self.logger.error(f"Error sending WebSocket update: {e}")
    
    async def _broadcast_update(self):
        """Broadcast updates to all connected WebSockets."""
        if not self.websockets:
            return
        
        try:
            data = await self._collect_all_metrics()
            message = json.dumps({
                'type': 'metrics_update',
                'data': data
            })
            
            # Send to all connected clients
            disconnected = set()
            for ws in self.websockets:
                try:
                    await ws.send_str(message)
                except Exception:
                    disconnected.add(ws)
            
            # Clean up disconnected clients
            self.websockets -= disconnected
        
        except Exception as e:
            self.logger.error(f"Error broadcasting update: {e}")
    
    async def _periodic_update_task(self):
        """Background task for periodic updates."""
        while True:
            try:
                await self._broadcast_update()
                await asyncio.sleep(10)  # Update every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic update task: {e}")
                await asyncio.sleep(10)
    
    def _get_uptime(self):
        """Get dashboard uptime."""
        # This is a simplified implementation
        return "N/A"
    
    def _generate_dashboard_html(self):
        """Generate the dashboard HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PG-Neo-Graph-RL Monitoring Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { color: #2c3e50; margin-bottom: 1rem; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }
        .metric { display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 4px; }
        .metric-value { font-weight: bold; color: #27ae60; }
        .alert { padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px; border-left: 4px solid; }
        .alert.warning { background: #fff3cd; border-color: #ffc107; color: #856404; }
        .alert.error { background: #f8d7da; border-color: #dc3545; color: #721c24; }
        .alert.critical { background: #f8d7da; border-color: #dc3545; color: #721c24; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-green { background: #27ae60; }
        .status-yellow { background: #f39c12; }
        .status-red { background: #e74c3c; }
        .task-list { max-height: 300px; overflow-y: auto; }
        .task-item { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; border-bottom: 1px solid #eee; }
        .task-item:last-child { border-bottom: none; }
        .btn { padding: 0.5rem 1rem; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .btn.small { padding: 0.25rem 0.5rem; font-size: 0.8rem; }
        .connection-status { position: fixed; top: 10px; right: 10px; padding: 0.5rem 1rem; border-radius: 4px; color: white; }
        .connected { background: #27ae60; }
        .disconnected { background: #e74c3c; }
        .loading { text-align: center; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 PG-Neo-Graph-RL Monitoring Dashboard</h1>
        <div id="connection-status" class="connection-status disconnected">Disconnected</div>
    </div>
    
    <div class="container">
        <div class="grid">
            <!-- System Metrics -->
            <div class="card">
                <h3>System Metrics</h3>
                <div id="system-metrics" class="loading">Loading...</div>
            </div>
            
            <!-- Application Metrics -->
            <div class="card">
                <h3>Application Metrics</h3>
                <div id="app-metrics" class="loading">Loading...</div>
            </div>
            
            <!-- Project Metrics -->
            <div class="card">
                <h3>Project Metrics</h3>
                <div id="project-metrics" class="loading">Loading...</div>
            </div>
            
            <!-- Alerts -->
            <div class="card">
                <h3>Active Alerts</h3>
                <div id="alerts" class="loading">Loading...</div>
            </div>
            
            <!-- Automation Tasks -->
            <div class="card">
                <h3>Automation Tasks</h3>
                <div id="automation-tasks" class="loading">Loading...</div>
            </div>
            
            <!-- Recent Activity -->
            <div class="card">
                <h3>Recent Activity</h3>
                <div id="recent-activity" class="loading">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        class Dashboard {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 1000;
                this.init();
            }
            
            init() {
                this.connect();
                this.setupEventListeners();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.updateConnectionStatus(true);
                    this.reconnectAttempts = 0;
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus(false);
                    this.scheduleReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectAttempts++;
                        console.log(`Reconnection attempt ${this.reconnectAttempts}`);
                        this.connect();
                    }, this.reconnectDelay * this.reconnectAttempts);
                }
            }
            
            updateConnectionStatus(connected) {
                const statusEl = document.getElementById('connection-status');
                if (connected) {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'connection-status connected';
                } else {
                    statusEl.textContent = 'Disconnected';
                    statusEl.className = 'connection-status disconnected';
                }
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'full_update':
                    case 'metrics_update':
                        this.updateMetrics(data.data);
                        break;
                    case 'automation_update':
                        this.updateAutomation(data.data);
                        break;
                    case 'task_result':
                        this.handleTaskResult(data);
                        break;
                    case 'task_error':
                        this.handleTaskError(data);
                        break;
                }
            }
            
            updateMetrics(data) {
                // System metrics
                const systemEl = document.getElementById('system-metrics');
                if (data.system_metrics) {
                    const sm = data.system_metrics;
                    systemEl.innerHTML = `
                        <div class="metric">
                            <span>CPU Usage:</span>
                            <span class="metric-value">${sm.cpu_usage?.toFixed(1) || 0}%</span>
                        </div>
                        <div class="metric">
                            <span>Memory Usage:</span>
                            <span class="metric-value">${sm.memory_usage?.toFixed(1) || 0}%</span>
                        </div>
                        <div class="metric">
                            <span>Disk Usage:</span>
                            <span class="metric-value">${sm.disk_usage?.toFixed(1) || 0}%</span>
                        </div>
                        <div class="metric">
                            <span>Processes:</span>
                            <span class="metric-value">${sm.process_count || 0}</span>
                        </div>
                    `;
                }
                
                // Application metrics
                const appEl = document.getElementById('app-metrics');
                if (data.app_metrics) {
                    const am = data.app_metrics;
                    appEl.innerHTML = `
                        <div class="metric">
                            <span>JAX Devices:</span>
                            <span class="metric-value">${am.jax_devices || 0} (${am.jax_platform || 'unknown'})</span>
                        </div>
                        <div class="metric">
                            <span>Training Metrics:</span>
                            <span class="metric-value">${Object.keys(am.training_metrics || {}).length}</span>
                        </div>
                        <div class="metric">
                            <span>Federated Metrics:</span>
                            <span class="metric-value">${Object.keys(am.federated_metrics || {}).length}</span>
                        </div>
                    `;
                }
                
                // Project metrics
                const projectEl = document.getElementById('project-metrics');
                if (data.project_metrics) {
                    const pm = data.project_metrics;
                    projectEl.innerHTML = `
                        <div class="metric">
                            <span>Code Coverage:</span>
                            <span class="metric-value">${pm.code_coverage?.toFixed(1) || 0}%</span>
                        </div>
                        <div class="metric">
                            <span>Test Results:</span>
                            <span class="metric-value">${(pm.test_count || 0) - (pm.failed_tests || 0)}/${pm.test_count || 0}</span>
                        </div>
                        <div class="metric">
                            <span>Code Quality:</span>
                            <span class="metric-value">${pm.code_quality_score?.toFixed(1) || 0}/100</span>
                        </div>
                        <div class="metric">
                            <span>Security Issues:</span>
                            <span class="metric-value">${pm.security_issues || 0}</span>
                        </div>
                        <div class="metric">
                            <span>Lines of Code:</span>
                            <span class="metric-value">${(pm.lines_of_code || 0).toLocaleString()}</span>
                        </div>
                    `;
                }
                
                // Alerts
                const alertsEl = document.getElementById('alerts');
                if (data.alerts && data.alerts.length > 0) {
                    alertsEl.innerHTML = data.alerts.map(alert => `
                        <div class="alert ${alert.type}">
                            <strong>${alert.category}:</strong> ${alert.message}
                        </div>
                    `).join('');
                } else {
                    alertsEl.innerHTML = '<div style="color: #27ae60; text-align: center;">✅ No active alerts</div>';
                }
            }
            
            updateAutomation(data) {
                const tasksEl = document.getElementById('automation-tasks');
                if (data.tasks) {
                    const tasks = Object.values(data.tasks);
                    tasksEl.innerHTML = `
                        <div class="task-list">
                            ${tasks.map(task => `
                                <div class="task-item">
                                    <div>
                                        <span class="status-indicator ${this.getStatusColor(task.last_status)}"></span>
                                        <strong>${task.name}</strong>
                                        ${task.currently_running ? ' (Running)' : ''}
                                        ${!task.enabled ? ' (Disabled)' : ''}
                                    </div>
                                    <button class="btn small" onclick="dashboard.runTask('${task.name}')" 
                                            ${task.currently_running ? 'disabled' : ''}>
                                        Run
                                    </button>
                                </div>
                            `).join('')}
                        </div>
                        <div style="margin-top: 1rem; font-size: 0.9em; color: #7f8c8d;">
                            Total: ${data.total_tasks}, Enabled: ${data.enabled_tasks}, Running: ${data.running_tasks}
                        </div>
                    `;
                }
                
                // Recent activity
                const activityEl = document.getElementById('recent-activity');
                if (data.recent_history) {
                    activityEl.innerHTML = data.recent_history.slice(0, 10).map(item => `
                        <div class="task-item">
                            <div>
                                <span class="status-indicator ${this.getStatusColor(item.status)}"></span>
                                <strong>${item.task_name}</strong>
                                <br><small>${new Date(item.start_time).toLocaleString()}</small>
                            </div>
                            <div style="text-align: right;">
                                <small>${item.execution_time?.toFixed(1) || 0}s</small>
                                ${item.error ? '<br><small style="color: #e74c3c;">Error</small>' : ''}
                            </div>
                        </div>
                    `).join('');
                }
            }
            
            getStatusColor(status) {
                switch (status) {
                    case 'completed': return 'status-green';
                    case 'running': return 'status-yellow';
                    case 'failed': return 'status-red';
                    default: return 'status-yellow';
                }
            }
            
            runTask(taskName) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: 'run_task',
                        task_name: taskName
                    }));
                }
            }
            
            handleTaskResult(data) {
                console.log(`Task ${data.task_name} completed:`, data.result);
                // Refresh automation data
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'get_automation' }));
                }
            }
            
            handleTaskError(data) {
                console.error(`Task ${data.task_name} failed:`, data.error);
                alert(`Task "${data.task_name}" failed: ${data.error}`);
            }
            
            setupEventListeners() {
                // Auto-refresh every 30 seconds
                setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({ type: 'get_metrics' }));
                        this.ws.send(JSON.stringify({ type: 'get_automation' }));
                    }
                }, 30000);
            }
        }
        
        // Initialize dashboard
        const dashboard = new Dashboard();
    </script>
</body>
</html>"""
    
    async def start_server(self):
        """Start the dashboard web server."""
        if not AIOHTTP_AVAILABLE:
            self.logger.error("aiohttp not available, cannot start web server")
            return
        
        # Create app
        app = await self.create_app()
        
        # Start background tasks
        self.background_tasks.append(
            asyncio.create_task(self._periodic_update_task())
        )
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"Dashboard started at http://{self.host}:{self.port}")
        
        try:
            # Keep server running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            self.logger.info("Shutting down dashboard...")
        finally:
            # Clean up
            for task in self.background_tasks:
                task.cancel()
            await runner.cleanup()


async def main():
    """Main entry point for monitoring dashboard."""
    parser = argparse.ArgumentParser(description="PG-Neo-Graph-RL Monitoring Dashboard")
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to'
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
        dashboard = MonitoringDashboard(args.host, args.port)
        await dashboard.start_server()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())