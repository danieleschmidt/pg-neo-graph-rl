#!/usr/bin/env python3
"""
Health check script for pg-neo-graph-rl containers.
Validates system health and readiness for production use.
"""

import sys
import time
import urllib.request
import urllib.error
import json
import subprocess
import psutil
from pathlib import Path


def check_http_endpoint(url: str, timeout: int = 5) -> bool:
    """Check if HTTP endpoint is responding."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def check_process_health() -> bool:
    """Check if main process is running and healthy."""
    try:
        # Check if Python process is running
        python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                          if p.info['name'] == 'python' and 'pg_neo_graph_rl' in ' '.join(p.info['cmdline'] or [])]
        
        if not python_processes:
            print("❌ No pg-neo-graph-rl processes found")
            return False
        
        # Check process health
        for proc in python_processes:
            try:
                proc_info = proc.as_dict(['pid', 'memory_percent', 'cpu_percent', 'status'])
                
                # Check if process is responsive
                if proc_info['status'] not in ['running', 'sleeping']:
                    print(f"❌ Process {proc_info['pid']} in bad state: {proc_info['status']}")
                    return False
                    
                # Check memory usage
                if proc_info['memory_percent'] > 95.0:
                    print(f"⚠️  Process {proc_info['pid']} using {proc_info['memory_percent']:.1f}% memory")
                    return False
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print("✅ Process health check passed")
        return True
        
    except Exception as e:
        print(f"❌ Process health check failed: {e}")
        return False


def check_system_resources() -> bool:
    """Check system resource availability."""
    try:
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            print(f"❌ Memory usage too high: {memory.percent:.1f}%")
            return False
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > 95:
            print(f"❌ Disk usage too high: {disk_percent:.1f}%")
            return False
        
        # Check if we have enough free memory for operation
        if memory.available < 100 * 1024 * 1024:  # 100MB minimum
            print(f"❌ Insufficient available memory: {memory.available / (1024*1024):.1f}MB")
            return False
        
        print("✅ System resources check passed")
        return True
        
    except Exception as e:
        print(f"❌ System resources check failed: {e}")
        return False


def check_file_system() -> bool:
    """Check critical file system paths."""
    try:
        critical_paths = [
            '/app',
            '/app/data',
            '/app/logs',
            '/app/checkpoints'
        ]
        
        for path_str in critical_paths:
            path = Path(path_str)
            if not path.exists():
                print(f"❌ Critical path missing: {path_str}")
                return False
            
            # Check if directory is writable
            if path.is_dir():
                test_file = path / '.health_check_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except (PermissionError, OSError):
                    print(f"❌ Path not writable: {path_str}")
                    return False
        
        print("✅ File system check passed")
        return True
        
    except Exception as e:
        print(f"❌ File system check failed: {e}")
        return False


def check_python_imports() -> bool:
    """Check if critical Python imports work."""
    try:
        # Test critical imports
        critical_imports = [
            'sys',
            'os',
            'time',
            'json',
            'threading',
        ]
        
        for module in critical_imports:
            try:
                __import__(module)
            except ImportError as e:
                print(f"❌ Failed to import {module}: {e}")
                return False
        
        # Test if we can import our package (without full initialization)
        try:
            import pg_neo_graph_rl
            print("✅ Package import successful")
        except ImportError as e:
            print(f"⚠️  Package import failed (may be expected in minimal container): {e}")
            # Don't fail health check for this in case JAX isn't available
        
        print("✅ Python imports check passed")
        return True
        
    except Exception as e:
        print(f"❌ Python imports check failed: {e}")
        return False


def check_configuration() -> bool:
    """Check if configuration files are present and valid."""
    try:
        config_files = [
            '/app/config/training_config.yaml',
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            if not path.exists():
                print(f"⚠️  Optional config file missing: {config_file}")
                continue
            
            # Basic validation - check if file is readable
            try:
                with open(path, 'r') as f:
                    content = f.read()
                if len(content) < 10:
                    print(f"❌ Config file too small: {config_file}")
                    return False
            except Exception as e:
                print(f"❌ Cannot read config file {config_file}: {e}")
                return False
        
        print("✅ Configuration check passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False


def run_health_checks(check_endpoints: bool = True) -> bool:
    """Run all health checks."""
    print("🏥 Starting health checks...")
    
    checks = [
        ("System Resources", check_system_resources),
        ("File System", check_file_system),
        ("Python Imports", check_python_imports),
        ("Configuration", check_configuration),
        ("Process Health", check_process_health),
    ]
    
    # Add endpoint checks if requested
    if check_endpoints:
        checks.append(("HTTP Endpoints", lambda: (
            check_http_endpoint("http://localhost:8080/health") and
            check_http_endpoint("http://localhost:8080/metrics")
        )))
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 Running {check_name} check...")
        try:
            result = check_func()
            results.append(result)
            if result:
                print(f"✅ {check_name}: PASS")
            else:
                print(f"❌ {check_name}: FAIL")
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Health Check Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All health checks passed - System is healthy")
        return True
    else:
        print("⚠️  Some health checks failed - System may have issues")
        return False


def main():
    """Main health check entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health check for pg-neo-graph-rl")
    parser.add_argument("--no-endpoints", action="store_true", 
                       help="Skip HTTP endpoint checks")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout for health checks in seconds")
    
    args = parser.parse_args()
    
    # Set timeout
    import signal
    def timeout_handler(signum, frame):
        print(f"\n⏰ Health check timed out after {args.timeout} seconds")
        sys.exit(1)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    try:
        # Run health checks
        healthy = run_health_checks(check_endpoints=not args.no_endpoints)
        
        # Exit with appropriate code
        if healthy:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Health check failed with exception: {e}")
        sys.exit(1)
    finally:
        signal.alarm(0)  # Cancel timeout


if __name__ == "__main__":
    main()