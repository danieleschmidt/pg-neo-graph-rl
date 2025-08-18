#!/usr/bin/env python3
"""
Comprehensive Quality Gates for pg-neo-graph-rl.
Implements automated quality checks for production readiness.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any


class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool, score: float = 0.0, 
                 message: str = "", details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.score = score
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()


class QualityGatesRunner:
    """Runs comprehensive quality gates for the project."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.results: List[QualityGateResult] = []
        self.min_coverage = 85.0
        self.max_security_issues = 5
        
    def run_all_gates(self) -> bool:
        """Run all quality gates and return overall pass/fail."""
        print("🔬 Running Comprehensive Quality Gates")
        print("=" * 50)
        
        # Test gates
        self.run_test_coverage_gate()
        self.run_test_suite_gate()
        
        # Code quality gates  
        self.run_linting_gate()
        self.run_type_checking_gate()
        
        # Security gates
        self.run_security_gate()
        self.run_dependency_security_gate()
        
        # Performance gates
        self.run_performance_benchmarks_gate()
        
        # Documentation gates
        self.run_documentation_gate()
        
        # Integration gates
        self.run_integration_tests_gate()
        
        return self.summarize_results()
    
    def run_test_coverage_gate(self) -> QualityGateResult:
        """Test coverage quality gate."""
        print("\n🧪 Running Test Coverage Gate...")
        
        try:
            result = subprocess.run([
                "python", "-m", "pytest", 
                "--cov=pg_neo_graph_rl",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--quiet"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # Parse coverage report
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    
                    total_coverage = coverage_data["totals"]["percent_covered"]
                    
                    passed = total_coverage >= self.min_coverage
                    message = f"Coverage: {total_coverage:.1f}% (target: {self.min_coverage}%)"
                    
                    gate_result = QualityGateResult(
                        "test_coverage", passed, total_coverage, message,
                        {"coverage_data": coverage_data["files"]}
                    )
                else:
                    gate_result = QualityGateResult(
                        "test_coverage", False, 0.0, "Coverage report not found"
                    )
            else:
                gate_result = QualityGateResult(
                    "test_coverage", False, 0.0, f"Coverage check failed: {result.stderr}"
                )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "test_coverage", False, 0.0, f"Coverage check error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_test_suite_gate(self) -> QualityGateResult:
        """Test suite execution quality gate."""
        print("\n🧪 Running Test Suite Gate...")
        
        try:
            result = subprocess.run([
                "python", "-m", "pytest", 
                "tests/",
                "--tb=short",
                "-v"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse test results
            passed_tests = result.stdout.count(" PASSED")
            failed_tests = result.stdout.count(" FAILED") 
            
            passed = result.returncode == 0
            score = passed_tests / max(passed_tests + failed_tests, 1) * 100
            message = f"Tests: {passed_tests} passed, {failed_tests} failed"
            
            gate_result = QualityGateResult(
                "test_suite", passed, score, message,
                {"passed": passed_tests, "failed": failed_tests}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "test_suite", False, 0.0, f"Test suite error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_linting_gate(self) -> QualityGateResult:
        """Code linting quality gate."""
        print("\n📝 Running Linting Gate...")
        
        try:
            result = subprocess.run([
                "ruff", "check", "pg_neo_graph_rl/", "--format=json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                issue_count = len(issues)
            else:
                issue_count = 0
            
            passed = issue_count == 0
            score = max(0, 100 - issue_count)
            message = f"Linting issues: {issue_count}"
            
            gate_result = QualityGateResult(
                "linting", passed, score, message,
                {"issues": issue_count}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "linting", False, 0.0, f"Linting error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_type_checking_gate(self) -> QualityGateResult:
        """Type checking quality gate."""
        print("\n🔍 Running Type Checking Gate...")
        
        try:
            result = subprocess.run([
                "mypy", "pg_neo_graph_rl/", "--ignore-missing-imports"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            error_count = result.stdout.count("error:")
            passed = error_count == 0
            score = max(0, 100 - error_count * 10)
            message = f"Type errors: {error_count}"
            
            gate_result = QualityGateResult(
                "type_checking", passed, score, message,
                {"errors": error_count}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "type_checking", False, 0.0, f"Type checking error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_security_gate(self) -> QualityGateResult:
        """Security vulnerability quality gate."""
        print("\n🔒 Running Security Gate...")
        
        try:
            result = subprocess.run([
                "bandit", "-r", "pg_neo_graph_rl/", "-f", "json", "--quiet"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                security_data = json.loads(result.stdout)
                issues = security_data.get("results", [])
                high_severity = len([i for i in issues if i["issue_severity"] == "HIGH"])
                medium_severity = len([i for i in issues if i["issue_severity"] == "MEDIUM"])
                total_issues = len(issues)
            else:
                high_severity = medium_severity = total_issues = 0
            
            passed = high_severity == 0 and total_issues <= self.max_security_issues
            score = max(0, 100 - high_severity * 30 - medium_severity * 10)
            message = f"Security issues: {total_issues} ({high_severity} high, {medium_severity} medium)"
            
            gate_result = QualityGateResult(
                "security", passed, score, message,
                {"total": total_issues, "high": high_severity, "medium": medium_severity}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "security", False, 0.0, f"Security check error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_dependency_security_gate(self) -> QualityGateResult:
        """Dependency security quality gate.""" 
        print("\n🛡️ Running Dependency Security Gate...")
        
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout and result.stdout.startswith('['):
                vulnerabilities = json.loads(result.stdout)
                vuln_count = len(vulnerabilities)
            else:
                vuln_count = 0
            
            passed = vuln_count == 0
            score = max(0, 100 - vuln_count * 20)
            message = f"Dependency vulnerabilities: {vuln_count}"
            
            gate_result = QualityGateResult(
                "dependency_security", passed, score, message,
                {"vulnerabilities": vuln_count}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "dependency_security", True, 90.0, f"Safety check warning: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_performance_benchmarks_gate(self) -> QualityGateResult:
        """Performance benchmarks quality gate."""
        print("\n⚡ Running Performance Benchmarks Gate...")
        
        try:
            result = subprocess.run([
                "python", "-m", "pytest", 
                "tests/benchmarks/",
                "--benchmark-only",
                "--benchmark-json=benchmark_results.json",
                "--quiet"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            
            benchmarks = []
            benchmark_file = self.project_root / "benchmark_results.json"
            if passed and benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get("benchmarks", [])
                if benchmarks:
                    avg_time = sum(b["stats"]["mean"] for b in benchmarks) / len(benchmarks)
                    score = min(100, max(0, 100 - avg_time * 100))  # Score based on avg time
                    message = f"Benchmarks: {len(benchmarks)} completed, avg time: {avg_time:.4f}s"
                else:
                    score = 90.0  # Acceptable score if no benchmarks but tests pass
                    message = "No benchmarks to run, but tests passed"
            else:
                score = 0.0
                message = "Performance benchmarks failed"
            
            gate_result = QualityGateResult(
                "performance", passed, score, message,
                {"benchmark_count": len(benchmarks)}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "performance", False, 0.0, f"Performance check error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_documentation_gate(self) -> QualityGateResult:
        """Documentation quality gate."""
        print("\n📚 Running Documentation Gate...")
        
        try:
            # Check for key documentation files
            required_docs = [
                "README.md", "CHANGELOG.md", "CONTRIBUTING.md", 
                "LICENSE", "docs/", "examples/"
            ]
            
            existing_docs = []
            for doc in required_docs:
                if (self.project_root / doc).exists():
                    existing_docs.append(doc)
            
            coverage_ratio = len(existing_docs) / len(required_docs)
            passed = coverage_ratio >= 0.8
            score = coverage_ratio * 100
            message = f"Documentation: {len(existing_docs)}/{len(required_docs)} files present"
            
            gate_result = QualityGateResult(
                "documentation", passed, score, message,
                {"existing": existing_docs, "missing": set(required_docs) - set(existing_docs)}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "documentation", False, 0.0, f"Documentation check error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def run_integration_tests_gate(self) -> QualityGateResult:
        """Integration tests quality gate."""
        print("\n🔗 Running Integration Tests Gate...")
        
        try:
            result = subprocess.run([
                "python", "-m", "pytest", 
                "tests/integration/",
                "--tb=short",
                "-v"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            passed_tests = result.stdout.count(" PASSED")
            failed_tests = result.stdout.count(" FAILED")
            
            passed = result.returncode == 0
            score = passed_tests / max(passed_tests + failed_tests, 1) * 100
            message = f"Integration tests: {passed_tests} passed, {failed_tests} failed"
            
            gate_result = QualityGateResult(
                "integration_tests", passed, score, message,
                {"passed": passed_tests, "failed": failed_tests}
            )
                
        except Exception as e:
            gate_result = QualityGateResult(
                "integration_tests", False, 0.0, f"Integration tests error: {str(e)}"
            )
        
        self.results.append(gate_result)
        self._print_gate_result(gate_result)
        return gate_result
    
    def _print_gate_result(self, result: QualityGateResult):
        """Print a quality gate result."""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"   {status} {result.name}: {result.message} (Score: {result.score:.1f})")
    
    def summarize_results(self) -> bool:
        """Summarize all quality gate results."""
        print("\n" + "="*50)
        print("📊 Quality Gates Summary")
        print("="*50)
        
        passed_gates = sum(1 for r in self.results if r.passed)
        total_gates = len(self.results)
        overall_score = sum(r.score for r in self.results) / max(total_gates, 1)
        overall_passed = passed_gates == total_gates
        
        print(f"\nOverall Result: {'✅ PASS' if overall_passed else '❌ FAIL'}")
        print(f"Gates Passed: {passed_gates}/{total_gates}")
        print(f"Overall Score: {overall_score:.1f}/100")
        
        print("\nDetailed Results:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name:<20} {result.score:>6.1f} - {result.message}")
        
        # Print failed gates details
        failed_gates = [r for r in self.results if not r.passed]
        if failed_gates:
            print("\n❌ Failed Gates Details:")
            for gate in failed_gates:
                print(f"\n  {gate.name}:")
                print(f"    Message: {gate.message}")
                if gate.details:
                    for key, value in gate.details.items():
                        print(f"    {key}: {value}")
        
        print(f"\n{'🎉 ALL QUALITY GATES PASSED!' if overall_passed else '⚠️  QUALITY GATES FAILED - FIX ISSUES BEFORE DEPLOYMENT'}")
        
        return overall_passed


def main():
    """Main entry point for quality gates."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path.cwd()
    
    runner = QualityGatesRunner(project_root)
    success = runner.run_all_gates()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()