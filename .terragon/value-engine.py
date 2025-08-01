#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Repository: pg-neo-graph-rl
Maturity: MATURING (50-75%)

Implements comprehensive value discovery, scoring, and autonomous execution
using WSJF + ICE + Technical Debt hybrid scoring model.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValueItem:
    """Represents a discovered value opportunity."""
    id: str
    title: str
    description: str
    category: str
    source: str
    files_affected: List[str]
    estimated_effort_hours: float
    
    # WSJF Components
    user_business_value: float  # 1-10
    time_criticality: float     # 1-10
    risk_reduction: float       # 1-10
    opportunity_enablement: float # 1-10
    
    # ICE Components  
    impact: float              # 1-10
    confidence: float          # 1-10  
    ease: float               # 1-10
    
    # Technical Debt
    debt_cost_hours: float     # Maintenance hours saved
    debt_interest_rate: float  # Future cost growth rate
    hotspot_multiplier: float  # 1-5x based on code churn/complexity
    
    # Metadata
    discovered_at: datetime
    priority_boost: float = 1.0  # Security/compliance multipliers
    risk_score: float = 0.0      # 0-1 execution risk
    
    def calculate_wsjf(self) -> float:
        """Calculate Weighted Shortest Job First score."""
        cost_of_delay = (
            self.user_business_value +
            self.time_criticality + 
            self.risk_reduction +
            self.opportunity_enablement
        )
        return cost_of_delay / max(self.estimated_effort_hours, 0.5)
    
    def calculate_ice(self) -> float:
        """Calculate Impact × Confidence × Ease score."""
        return self.impact * self.confidence * self.ease
    
    def calculate_technical_debt_score(self) -> float:
        """Calculate technical debt reduction value."""
        debt_value = (self.debt_cost_hours + self.debt_interest_rate * 10)
        return debt_value * self.hotspot_multiplier
    
    def calculate_composite_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite value score."""
        wsjf = self.calculate_wsjf()
        ice = self.calculate_ice() 
        debt = self.calculate_technical_debt_score()
        
        # Normalize scores to 0-100 scale
        wsjf_norm = min(wsjf * 2, 100)  # WSJF typically 0-50
        ice_norm = min(ice / 10, 100)   # ICE typically 0-1000
        debt_norm = min(debt / 5, 100)  # Debt typically 0-500
        
        composite = (
            weights['wsjf'] * wsjf_norm +
            weights['ice'] * ice_norm + 
            weights['technicalDebt'] * debt_norm
        )
        
        # Apply priority boost for security/compliance
        return composite * self.priority_boost

class ValueDiscoveryEngine:
    """Discovers, scores, and prioritizes value opportunities."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        """Initialize the value discovery engine."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.discovered_items: List[ValueItem] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            # Simple YAML parsing without external dependencies
            with open(self.config_path, 'r') as f:
                content = f.read()
                # Basic YAML parsing for our config structure
                config = {}
                current_section = None
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if ':' in line and not line.startswith(' '):
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if value:
                                try:
                                    config[key] = float(value) if '.' in value else int(value)
                                except ValueError:
                                    config[key] = value.strip('"\'')
                            else:
                                config[key] = {}
                                current_section = config[key]
                        elif current_section is not None and line.startswith(' ') and ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if value:
                                try:
                                    current_section[key] = float(value) if '.' in value else int(value)
                                except ValueError:
                                    current_section[key] = value.strip('"\'')
                
                # Set nested structure for our config
                return {
                    'repository': {'name': 'pg-neo-graph-rl', 'maturity_level': 'maturing', 'maturity_score': 65},
                    'scoring': {
                        'weights': {'wsjf': 0.6, 'ice': 0.1, 'technicalDebt': 0.2, 'security': 0.1},
                        'thresholds': {'min_execution_score': 15.0, 'max_risk_tolerance': 0.8}
                    },
                    'discovery': {
                        'sources': {
                            'enabled': {
                                'git_history': True, 'static_analysis': True, 'dependency_scan': True,
                                'code_comments': True, 'test_coverage': True
                            }
                        }
                    }
                }
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'scoring': {
                'weights': {'wsjf': 0.6, 'ice': 0.1, 'technicalDebt': 0.2, 'security': 0.1},
                'thresholds': {'min_execution_score': 15.0}
            },
            'discovery': {'sources': {'enabled': {'git_history': True, 'static_analysis': True}}}
        }
    
    def discover_all_value_items(self) -> List[ValueItem]:
        """Execute comprehensive value discovery across all sources."""
        logger.info("Starting comprehensive value discovery...")
        
        # Clear previous discoveries
        self.discovered_items = []
        
        # Discovery sources
        if self.config['discovery']['sources']['enabled'].get('git_history', False):
            self.discovered_items.extend(self._discover_from_git_history())
            
        if self.config['discovery']['sources']['enabled'].get('static_analysis', False):
            self.discovered_items.extend(self._discover_from_static_analysis())
            
        if self.config['discovery']['sources']['enabled'].get('dependency_scan', False):
            self.discovered_items.extend(self._discover_from_dependencies())
            
        if self.config['discovery']['sources']['enabled'].get('code_comments', False):
            self.discovered_items.extend(self._discover_from_code_comments())
            
        if self.config['discovery']['sources']['enabled'].get('test_coverage', False):
            self.discovered_items.extend(self._discover_from_test_gaps())
        
        # Score and rank all items
        self._score_and_rank_items()
        
        logger.info(f"Discovered {len(self.discovered_items)} value opportunities")
        return self.discovered_items
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover opportunities from Git commit history and patterns."""
        items = []
        
        try:
            # Find TODOs, FIXMEs in commit messages
            result = subprocess.run(
                ['git', 'log', '--grep=TODO', '--grep=FIXME', '--grep=HACK', '--oneline', '-n', '50'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash = line.split()[0]
                    commit_msg = ' '.join(line.split()[1:])
                    
                    items.append(ValueItem(
                        id=f"git-todo-{commit_hash[:8]}",
                        title=f"Address TODO in commit: {commit_msg[:50]}...",
                        description=f"Commit {commit_hash} contains TODO/FIXME markers",
                        category="technical_debt",
                        source="git_history",
                        files_affected=[],
                        estimated_effort_hours=2.0,
                        user_business_value=4.0,
                        time_criticality=3.0,
                        risk_reduction=5.0,
                        opportunity_enablement=3.0,
                        impact=6.0,
                        confidence=8.0,
                        ease=7.0,
                        debt_cost_hours=3.0,
                        debt_interest_rate=0.1,
                        hotspot_multiplier=1.5,
                        discovered_at=datetime.now()
                    ))
            
            # Identify high-churn files that need refactoring
            result = subprocess.run(
                ['git', 'log', '--pretty=format:', '--name-only', '--since=3.months.ago'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                file_changes = {}
                for line in result.stdout.strip().split('\n'):
                    if line and line.endswith('.py'):
                        file_changes[line] = file_changes.get(line, 0) + 1
                
                # High-churn files (>10 changes in 3 months)
                for file_path, change_count in file_changes.items():
                    if change_count > 10:
                        items.append(ValueItem(
                            id=f"refactor-{file_path.replace('/', '-')}",
                            title=f"Refactor high-churn file: {file_path}",
                            description=f"File changed {change_count} times in 3 months, needs refactoring",
                            category="refactoring",
                            source="git_history",
                            files_affected=[file_path],
                            estimated_effort_hours=min(change_count * 0.5, 8.0),
                            user_business_value=7.0,
                            time_criticality=5.0,
                            risk_reduction=8.0,
                            opportunity_enablement=6.0,
                            impact=8.0,
                            confidence=7.0,
                            ease=5.0,
                            debt_cost_hours=change_count * 0.3,
                            debt_interest_rate=0.15,
                            hotspot_multiplier=3.0,
                            discovered_at=datetime.now()
                        ))
                        
        except subprocess.SubprocessError as e:
            logger.warning(f"Git history analysis failed: {e}")
            
        return items
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover opportunities from static analysis tools."""
        items = []
        
        # Missing source code implementation
        if not any(Path(self.repo_root).glob("**/*.py")):
            items.append(ValueItem(
                id="impl-core-001", 
                title="Implement core pg_neo_graph_rl package structure",
                description="No source code exists - need to implement the complete library",
                category="implementation",
                source="static_analysis",
                files_affected=["pg_neo_graph_rl/__init__.py"],
                estimated_effort_hours=40.0,
                user_business_value=10.0,
                time_criticality=9.0,
                risk_reduction=8.0,
                opportunity_enablement=10.0,
                impact=10.0,
                confidence=9.0,
                ease=6.0,
                debt_cost_hours=50.0,
                debt_interest_rate=0.2,
                hotspot_multiplier=4.0,
                discovered_at=datetime.now(),
                priority_boost=2.0  # Critical for functionality
            ))
            
            # Break down implementation into modules
            modules = [
                ("core", "Core federated learning engine", 12.0),
                ("algorithms", "Graph RL algorithms (PPO, SAC)", 16.0),
                ("environments", "Environment interfaces and implementations", 10.0), 
                ("networks", "Graph neural network architectures", 14.0),
                ("communication", "Inter-agent communication protocols", 8.0),
                ("monitoring", "Metrics collection and visualization", 6.0)
            ]
            
            for module, desc, effort in modules:
                items.append(ValueItem(
                    id=f"impl-{module}-001",
                    title=f"Implement {module} module",
                    description=desc,
                    category="implementation",
                    source="static_analysis", 
                    files_affected=[f"pg_neo_graph_rl/{module}/__init__.py"],
                    estimated_effort_hours=effort,
                    user_business_value=9.0,
                    time_criticality=8.0,
                    risk_reduction=7.0,
                    opportunity_enablement=9.0,
                    impact=9.0,
                    confidence=8.0,
                    ease=7.0,
                    debt_cost_hours=effort * 0.8,
                    debt_interest_rate=0.15,
                    hotspot_multiplier=2.5,
                    discovered_at=datetime.now(),
                    priority_boost=1.8
                ))
        
        # Check for missing .gitignore patterns
        gitignore_path = self.repo_root / ".gitignore"
        if not gitignore_path.exists():
            items.append(ValueItem(
                id="setup-gitignore-001",
                title="Create comprehensive .gitignore file",
                description="Missing .gitignore file for Python/JAX project",
                category="setup",
                source="static_analysis",
                files_affected=[".gitignore"],
                estimated_effort_hours=0.5,
                user_business_value=6.0,
                time_criticality=4.0,
                risk_reduction=7.0,
                opportunity_enablement=5.0,
                impact=7.0,
                confidence=10.0,
                ease=10.0,
                debt_cost_hours=2.0,
                debt_interest_rate=0.05,
                hotspot_multiplier=1.0,
                discovered_at=datetime.now()
            ))
        
        return items
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover dependency-related opportunities."""
        items = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                
                for pkg in outdated[:5]:  # Top 5 outdated packages
                    is_security = 'security' in pkg.get('name', '').lower()
                    priority = 2.0 if is_security else 1.0
                    
                    items.append(ValueItem(
                        id=f"dep-update-{pkg['name']}",
                        title=f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        description=f"Dependency update for {pkg['name']}",
                        category="dependency_update",
                        source="dependency_scan",
                        files_affected=["requirements.txt", "pyproject.toml"],
                        estimated_effort_hours=1.0,
                        user_business_value=5.0,
                        time_criticality=6.0 if is_security else 3.0,
                        risk_reduction=8.0 if is_security else 4.0,
                        opportunity_enablement=4.0,
                        impact=6.0 if is_security else 4.0,
                        confidence=9.0,
                        ease=8.0,
                        debt_cost_hours=2.0,
                        debt_interest_rate=0.1,
                        hotspot_multiplier=1.0,
                        discovered_at=datetime.now(),
                        priority_boost=priority
                    ))
                    
        except (subprocess.SubprocessError, json.JSONDecodeError):
            logger.warning("Dependency analysis failed")
            
        return items
    
    def _discover_from_code_comments(self) -> List[ValueItem]:
        """Discover opportunities from TODO/FIXME comments in code."""
        items = []
        
        try:
            # Search for TODO/FIXME/HACK comments
            result = subprocess.run(
                ['grep', '-r', '-n', '-i', '--include=*.py', 
                 '-E', '(TODO|FIXME|HACK|DEPRECATED)', '.'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            for line in result.stdout.strip().split('\n'):
                if line and ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path, line_num, comment = parts
                        
                        # Extract comment text
                        comment_text = re.sub(r'^\s*#\s*(TODO|FIXME|HACK|DEPRECATED)[:\s]*', '', comment, flags=re.IGNORECASE)
                        
                        items.append(ValueItem(
                            id=f"comment-{file_path.replace('/', '-')}-{line_num}",
                            title=f"Address TODO in {file_path}:{line_num}",
                            description=comment_text[:100],
                            category="technical_debt",
                            source="code_comments", 
                            files_affected=[file_path],
                            estimated_effort_hours=1.5,
                            user_business_value=5.0,
                            time_criticality=4.0,
                            risk_reduction=6.0,
                            opportunity_enablement=4.0,
                            impact=6.0,
                            confidence=7.0,
                            ease=8.0,
                            debt_cost_hours=2.0,
                            debt_interest_rate=0.08,
                            hotspot_multiplier=1.2,
                            discovered_at=datetime.now()
                        ))
                        
        except subprocess.SubprocessError:
            # grep returns non-zero when no matches found
            pass
            
        return items
    
    def _discover_from_test_gaps(self) -> List[ValueItem]:
        """Discover testing-related opportunities."""
        items = []
        
        # Low test coverage areas (simulated - would use coverage.py in real implementation)
        test_files = list(Path(self.repo_root).glob("tests/**/*.py"))
        source_files = list(Path(self.repo_root).glob("pg_neo_graph_rl/**/*.py"))
        
        if len(source_files) == 0 and len(test_files) > 0:
            items.append(ValueItem(
                id="test-refactor-001",
                title="Align test structure with missing source code",
                description="Tests exist but source code is missing - need to implement or refactor tests",
                category="testing",
                source="test_coverage",
                files_affected=[str(f) for f in test_files],
                estimated_effort_hours=6.0,
                user_business_value=7.0,
                time_criticality=6.0,
                risk_reduction=8.0,
                opportunity_enablement=7.0,
                impact=8.0,
                confidence=9.0,
                ease=6.0,
                debt_cost_hours=10.0,
                debt_interest_rate=0.12,
                hotspot_multiplier=2.0,
                discovered_at=datetime.now()
            ))
        
        return items
    
    def _score_and_rank_items(self):
        """Score and rank all discovered items by composite value."""
        weights = self.config['scoring']['weights']
        
        for item in self.discovered_items:
            item.composite_score = item.calculate_composite_score(weights)
        
        # Sort by composite score (descending)
        self.discovered_items.sort(key=lambda x: x.composite_score, reverse=True)
    
    def get_next_best_value_item(self) -> Optional[ValueItem]:
        """Get the highest-value item that meets execution criteria."""
        min_score = self.config['scoring']['thresholds']['min_execution_score']
        
        for item in self.discovered_items:
            if (item.composite_score >= min_score and 
                item.risk_score <= self.config['scoring']['thresholds'].get('max_risk_tolerance', 0.8)):
                return item
                
        return None
    
    def generate_backlog_report(self) -> str:
        """Generate comprehensive backlog report in Markdown."""
        report = f"""# 📊 Autonomous Value Backlog

**Repository**: {self.config['repository']['name']}
**Maturity Level**: {self.config['repository']['maturity_level'].upper()} ({self.config['repository']['maturity_score']}%)
**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Value Items**: {len(self.discovered_items)}

## 🎯 Next Best Value Item

"""
        
        next_item = self.get_next_best_value_item()
        if next_item:
            report += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.calculate_wsjf():.1f} | **ICE**: {next_item.calculate_ice():.0f} | **Tech Debt**: {next_item.calculate_technical_debt_score():.1f}
- **Estimated Effort**: {next_item.estimated_effort_hours:.1f} hours
- **Category**: {next_item.category}
- **Expected Impact**: {next_item.description}

"""
        else:
            report += "No items meet execution criteria.\n\n"
        
        report += """## 📋 Top 15 Backlog Items

| Rank | ID | Title | Score | Category | Effort (hrs) | Source |
|------|-----|-------|-------|----------|--------------|--------|
"""
        
        for i, item in enumerate(self.discovered_items[:15], 1):
            report += f"| {i} | {item.id} | {item.title[:50]}{'...' if len(item.title) > 50 else ''} | {item.composite_score:.1f} | {item.category} | {item.estimated_effort_hours:.1f} | {item.source} |\n"
        
        report += f"""

## 📈 Value Metrics

- **Average Item Score**: {sum(item.composite_score for item in self.discovered_items) / len(self.discovered_items):.1f}
- **High-Value Items (>50 score)**: {sum(1 for item in self.discovered_items if item.composite_score > 50)}
- **Total Estimated Effort**: {sum(item.estimated_effort_hours for item in self.discovered_items):.1f} hours
- **Potential Debt Reduction**: {sum(item.debt_cost_hours for item in self.discovered_items):.1f} hours

## 🔄 Discovery Sources Distribution

"""
        
        sources = {}
        for item in self.discovered_items:
            sources[item.source] = sources.get(item.source, 0) + 1
            
        for source, count in sorted(sources.items()):
            percentage = (count / len(self.discovered_items)) * 100
            report += f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        report += f"""

## 📊 Category Breakdown

"""
        
        categories = {}
        for item in self.discovered_items:
            categories[item.category] = categories.get(item.category, 0) + 1
            
        for category, count in sorted(categories.items()):
            percentage = (count / len(self.discovered_items)) * 100
            total_effort = sum(item.estimated_effort_hours for item in self.discovered_items if item.category == category)
            report += f"- **{category.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%) - {total_effort:.1f} hours\n"
        
        return report

def main():
    """Main execution function for value discovery."""
    engine = ValueDiscoveryEngine()
    
    # Discover all value opportunities  
    items = engine.discover_all_value_items()
    
    # Generate and save backlog report
    report = engine.generate_backlog_report()
    
    # Save to BACKLOG.md
    backlog_path = Path("BACKLOG.md")
    with open(backlog_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Discovered {len(items)} value opportunities")
    print(f"📄 Backlog report saved to {backlog_path}")
    
    # Print next best item
    next_item = engine.get_next_best_value_item()
    if next_item:
        print(f"🎯 Next best value item: {next_item.title} (Score: {next_item.composite_score:.1f})")
    else:
        print("⚠️  No items meet execution criteria")

if __name__ == "__main__":
    main()