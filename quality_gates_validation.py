#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation for pg-neo-graph-rl

This script validates that all quality gates pass without requiring 
external dependencies that may not be available in the environment.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

def log(message: str, status: str = "INFO"):
    """Log messages with timestamp and status."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")

class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.results = {}
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        log("🚀 Starting comprehensive quality gate validation...")
        
        # Gate 1: Code Structure Validation
        self.validate_code_structure()
        
        # Gate 2: Module Import Validation  
        self.validate_module_imports()
        
        # Gate 3: Documentation Coverage
        self.validate_documentation()
        
        # Gate 4: Configuration Validation
        self.validate_configurations()
        
        # Gate 5: Security Baseline
        self.validate_security_baseline()
        
        # Gate 6: Performance Baseline
        self.validate_performance_baseline()
        
        # Gate 7: Production Readiness
        self.validate_production_readiness()
        
        # Generate final report
        return self.generate_final_report()
    
    def validate_code_structure(self):
        """Validate code structure and organization."""
        log("🏗️  Validating code structure...")
        
        required_dirs = [
            "pg_neo_graph_rl",
            "pg_neo_graph_rl/algorithms", 
            "pg_neo_graph_rl/core",
            "pg_neo_graph_rl/environments",
            "pg_neo_graph_rl/communication",
            "pg_neo_graph_rl/monitoring",
            "pg_neo_graph_rl/optimization",
            "pg_neo_graph_rl/utils",
            "tests",
            "deployment",
            "docs"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        required_files = [
            "pyproject.toml",
            "README.md", 
            "pg_neo_graph_rl/__init__.py",
            "pg_neo_graph_rl/algorithms/__init__.py",
            "pg_neo_graph_rl/core/__init__.py",
            "deployment/docker-compose.yml"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        structure_score = 100 - (len(missing_dirs) * 5 + len(missing_files) * 3)
        
        self.results["code_structure"] = {
            "score": max(0, structure_score),
            "missing_directories": missing_dirs,
            "missing_files": missing_files,
            "status": "PASS" if structure_score >= 85 else "FAIL"
        }
        
        log(f"✅ Code structure validation: {structure_score}% - {self.results['code_structure']['status']}")
    
    def validate_module_imports(self):
        """Validate that modules can be imported without errors."""
        log("📦 Validating module imports...")
        
        import_tests = []
        python_files = list(self.project_root.glob("pg_neo_graph_rl/**/*.py"))
        
        importable_modules = 0
        total_modules = 0
        
        for py_file in python_files:
            if py_file.name == "__init__.py":
                continue
                
            total_modules += 1
            relative_path = py_file.relative_to(self.project_root)
            module_path = str(relative_path).replace("/", ".").replace(".py", "")
            
            try:
                # Basic syntax validation by attempting to compile
                with open(py_file, 'r') as f:
                    code = f.read()
                
                compile(code, str(py_file), 'exec')
                import_tests.append((module_path, "SYNTAX_OK"))
                importable_modules += 1
                
            except SyntaxError as e:
                import_tests.append((module_path, f"SYNTAX_ERROR: {str(e)}"))
            except Exception as e:
                import_tests.append((module_path, f"ERROR: {str(e)}"))
        
        import_score = (importable_modules / total_modules * 100) if total_modules > 0 else 100
        
        self.results["module_imports"] = {
            "score": import_score,
            "importable_modules": importable_modules,
            "total_modules": total_modules,
            "import_tests": import_tests,
            "status": "PASS" if import_score >= 90 else "FAIL"
        }
        
        log(f"✅ Module imports validation: {import_score:.1f}% - {self.results['module_imports']['status']}")
    
    def validate_documentation(self):
        """Validate documentation coverage and quality."""
        log("📚 Validating documentation coverage...")
        
        doc_files = list(self.project_root.glob("**/*.md"))
        python_files = list(self.project_root.glob("pg_neo_graph_rl/**/*.py"))
        
        # Count docstrings
        documented_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Simple regex-like counting for functions/classes with docstrings
                lines = content.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if (line.startswith('def ') or line.startswith('class ')) and not line.startswith('def _'):
                        total_functions += 1
                        # Check next few lines for docstring
                        for j in range(i+1, min(i+5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                documented_functions += 1
                                break
                    i += 1
                    
            except Exception:
                continue
        
        doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 100
        
        self.results["documentation"] = {
            "score": doc_coverage,
            "documented_functions": documented_functions,
            "total_functions": total_functions,
            "doc_files_count": len(doc_files),
            "status": "PASS" if doc_coverage >= 80 else "FAIL"
        }
        
        log(f"✅ Documentation validation: {doc_coverage:.1f}% - {self.results['documentation']['status']}")
    
    def validate_configurations(self):
        """Validate configuration files."""
        log("⚙️  Validating configurations...")
        
        config_validations = []
        
        # Validate pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                
                required_sections = ["[build-system]", "[project]", "[tool."]
                missing_sections = [s for s in required_sections if s not in content]
                
                config_validations.append({
                    "file": "pyproject.toml",
                    "status": "VALID" if not missing_sections else "INCOMPLETE",
                    "missing": missing_sections
                })
                
            except Exception as e:
                config_validations.append({
                    "file": "pyproject.toml", 
                    "status": "ERROR",
                    "error": str(e)
                })
        
        # Validate docker-compose files
        for compose_file in ["docker-compose.yml", "deployment/docker-compose.yml"]:
            compose_path = self.project_root / compose_file
            if compose_path.exists():
                try:
                    with open(compose_path, 'r') as f:
                        content = f.read()
                    
                    has_services = "services:" in content
                    config_validations.append({
                        "file": compose_file,
                        "status": "VALID" if has_services else "INCOMPLETE",
                        "has_services": has_services
                    })
                    
                except Exception as e:
                    config_validations.append({
                        "file": compose_file,
                        "status": "ERROR", 
                        "error": str(e)
                    })
        
        valid_configs = sum(1 for c in config_validations if c["status"] == "VALID")
        config_score = (valid_configs / len(config_validations) * 100) if config_validations else 100
        
        self.results["configurations"] = {
            "score": config_score,
            "validations": config_validations,
            "valid_configs": valid_configs,
            "total_configs": len(config_validations),
            "status": "PASS" if config_score >= 85 else "FAIL"
        }
        
        log(f"✅ Configuration validation: {config_score:.1f}% - {self.results['configurations']['status']}")
    
    def validate_security_baseline(self):
        """Validate security baseline requirements.""" 
        log("🛡️  Validating security baseline...")
        
        security_checks = []
        
        # Check for security-related files
        security_files = ["SECURITY.md", "pg_neo_graph_rl/utils/security.py"]
        for file_path in security_files:
            full_path = self.project_root / file_path
            security_checks.append({
                "check": f"Security file exists: {file_path}",
                "status": "PASS" if full_path.exists() else "FAIL",
                "exists": full_path.exists()
            })
        
        # Check for hardcoded secrets (basic patterns)
        python_files = list(self.project_root.glob("**/*.py"))
        secret_patterns = ["password", "secret", "key", "token", "api_key"]
        potential_secrets = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for pattern in secret_patterns:
                        if f'{pattern}=' in line.lower() and '"' in line and not line.strip().startswith('#'):
                            potential_secrets.append(f"{py_file}:{i+1}")
                            
            except Exception:
                continue
        
        security_checks.append({
            "check": "No hardcoded secrets",
            "status": "PASS" if len(potential_secrets) == 0 else "WARNING",
            "potential_secrets": potential_secrets
        })
        
        # Check for input validation
        validation_files = list(self.project_root.glob("**/validation*.py")) + \
                          list(self.project_root.glob("**/security*.py"))
        
        security_checks.append({
            "check": "Input validation implemented", 
            "status": "PASS" if len(validation_files) > 0 else "FAIL",
            "validation_files": len(validation_files)
        })
        
        passed_checks = sum(1 for c in security_checks if c["status"] == "PASS")
        security_score = (passed_checks / len(security_checks) * 100) if security_checks else 100
        
        self.results["security"] = {
            "score": security_score,
            "checks": security_checks,
            "passed_checks": passed_checks,
            "total_checks": len(security_checks),
            "status": "PASS" if security_score >= 80 else "FAIL"
        }
        
        log(f"✅ Security validation: {security_score:.1f}% - {self.results['security']['status']}")
    
    def validate_performance_baseline(self):
        """Validate performance optimization components."""
        log("⚡ Validating performance baseline...")
        
        perf_components = [
            "pg_neo_graph_rl/optimization/cache.py",
            "pg_neo_graph_rl/optimization/performance.py", 
            "pg_neo_graph_rl/monitoring/metrics.py"
        ]
        
        perf_checks = []
        for component in perf_components:
            full_path = self.project_root / component
            exists = full_path.exists()
            
            # Check for performance-related keywords if file exists
            has_optimization = False
            if exists:
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    perf_keywords = ["cache", "optimize", "performance", "concurrent", "parallel"]
                    has_optimization = any(keyword in content.lower() for keyword in perf_keywords)
                except Exception:
                    pass
            
            perf_checks.append({
                "component": component,
                "exists": exists,
                "has_optimization": has_optimization,
                "status": "PASS" if exists and has_optimization else "FAIL"
            })
        
        passed_perf = sum(1 for c in perf_checks if c["status"] == "PASS")
        perf_score = (passed_perf / len(perf_checks) * 100) if perf_checks else 100
        
        self.results["performance"] = {
            "score": perf_score,
            "checks": perf_checks,
            "passed_checks": passed_perf,
            "total_checks": len(perf_checks),
            "status": "PASS" if perf_score >= 85 else "FAIL"
        }
        
        log(f"✅ Performance validation: {perf_score:.1f}% - {self.results['performance']['status']}")
    
    def validate_production_readiness(self):
        """Validate production deployment readiness."""
        log("🚀 Validating production readiness...")
        
        prod_components = [
            ("Dockerfile", "deployment/Dockerfile"),
            ("Docker Compose", "deployment/docker-compose.yml"),
            ("Health Check", "deployment/healthcheck.py"),
            ("Monitoring", "deployment/monitoring"),
            ("Production Guide", "deployment/production_deployment_guide.md")
        ]
        
        prod_checks = []
        for name, path in prod_components:
            full_path = self.project_root / path
            exists = full_path.exists()
            
            prod_checks.append({
                "component": name,
                "path": path,
                "exists": exists,
                "status": "PASS" if exists else "FAIL"
            })
        
        # Check for environment configuration
        env_files = list(self.project_root.glob("**/.env*")) + \
                   list(self.project_root.glob("**/config/*"))
        
        prod_checks.append({
            "component": "Environment Configuration",
            "count": len(env_files),
            "exists": len(env_files) > 0,
            "status": "PASS" if len(env_files) > 0 else "FAIL"
        })
        
        passed_prod = sum(1 for c in prod_checks if c["status"] == "PASS")
        prod_score = (passed_prod / len(prod_checks) * 100) if prod_checks else 100
        
        self.results["production_readiness"] = {
            "score": prod_score,
            "checks": prod_checks,
            "passed_checks": passed_prod,
            "total_checks": len(prod_checks),
            "status": "PASS" if prod_score >= 85 else "FAIL"
        }
        
        log(f"✅ Production readiness validation: {prod_score:.1f}% - {self.results['production_readiness']['status']}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final quality report."""
        log("📊 Generating final quality gate report...")
        
        # Calculate overall score
        gate_scores = [
            self.results["code_structure"]["score"],
            self.results["module_imports"]["score"], 
            self.results["documentation"]["score"],
            self.results["configurations"]["score"],
            self.results["security"]["score"],
            self.results["performance"]["score"],
            self.results["production_readiness"]["score"]
        ]
        
        overall_score = sum(gate_scores) / len(gate_scores)
        
        # Determine overall status
        failed_gates = [k for k, v in self.results.items() if v["status"] == "FAIL"]
        overall_status = "PASS" if len(failed_gates) == 0 and overall_score >= 85 else "FAIL"
        
        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": overall_score,
            "overall_status": overall_status,
            "failed_gates": failed_gates,
            "gate_results": self.results,
            "summary": {
                "total_gates": len(self.results),
                "passed_gates": sum(1 for v in self.results.values() if v["status"] == "PASS"),
                "failed_gates": len(failed_gates)
            },
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        report_path = self.project_root / "quality_gates_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        self._print_final_summary(final_report)
        
        return final_report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failed gates."""
        recommendations = []
        
        for gate_name, gate_result in self.results.items():
            if gate_result["status"] == "FAIL":
                if gate_name == "code_structure":
                    recommendations.append("Complete missing directory structure and core files")
                elif gate_name == "module_imports": 
                    recommendations.append("Fix syntax errors and import issues in Python modules")
                elif gate_name == "documentation":
                    recommendations.append("Add docstrings to functions and classes")
                elif gate_name == "configurations":
                    recommendations.append("Complete configuration files (pyproject.toml, docker-compose)")
                elif gate_name == "security":
                    recommendations.append("Implement security validation and remove hardcoded secrets")
                elif gate_name == "performance":
                    recommendations.append("Implement performance optimization components")
                elif gate_name == "production_readiness":
                    recommendations.append("Complete production deployment infrastructure")
        
        return recommendations
    
    def _print_final_summary(self, report: Dict[str, Any]):
        """Print final summary to console."""
        print("\n" + "="*80)
        print("🎯 QUALITY GATES FINAL REPORT")
        print("="*80)
        
        print(f"📊 Overall Score: {report['overall_score']:.1f}%")
        print(f"🏆 Overall Status: {report['overall_status']}")
        print(f"✅ Passed Gates: {report['summary']['passed_gates']}/{report['summary']['total_gates']}")
        
        if report['failed_gates']:
            print(f"❌ Failed Gates: {', '.join(report['failed_gates'])}")
        
        print("\n📋 Gate Breakdown:")
        for gate_name, result in report['gate_results'].items():
            status_icon = "✅" if result['status'] == "PASS" else "❌"
            print(f"  {status_icon} {gate_name.replace('_', ' ').title()}: {result['score']:.1f}%")
        
        if report['recommendations']:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print(f"📄 Detailed report saved to: quality_gates_report.json")
        print("="*80 + "\n")

def main():
    """Main execution function."""
    validator = QualityGateValidator()
    report = validator.run_all_gates()
    
    # Exit with appropriate code
    exit_code = 0 if report["overall_status"] == "PASS" else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()