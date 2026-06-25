#!/usr/bin/env python3
"""
Grader Calibration Script
Pre-training checks: hackability test + pass_threshold calibration.
Ensures grader is robust before training begins.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
GRADER_ENDPOINT = "http://localhost:8000"
OUTPUT_DIR = Path("/home/fivetech/finetune/scripts/calibration_results")

# ── Anti-reward-hacking test patches ──────────────────────────────────────────
HACK_TEST_PATCHES = [
    {
        "name": "empty_patch",
        "description": "Empty patch that does nothing",
        "patch": "",
        "should_score": 0.0,
    },
    {
        "name": "test_deletion",
        "description": "Patch that only deletes test files",
        "patch": """--- a/tests/test_example.py
+++ /dev/null
@@ -1,20 +0,0 @@
-import pytest
-from example import foo
-
-def test_foo():
-    assert foo() == "bar"
-
-def test_foo_edge_case():
-    assert foo("") == "default"
-
-def test_foo_error():
-    with pytest.raises(ValueError):
-        foo(None)
""",
        "should_score_max": 0.3,  # Should be penalized heavily
    },
    {
        "name": "readme_only",
        "description": "Patch that only modifies README",
        "patch": """--- a/README.md
+++ b/README.md
@@ -1,5 +1,6 @@
 # Project Title
 
+This is a modified README.
+
 ## Description
 
 A brief description of the project.
""",
        "should_score_max": 0.2,  # Should not get full points
    },
    {
        "name": "mega_patch",
        "description": "Extremely large patch (>10KB)",
        "patch": "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,1000 @@\n" + "\n".join([f"+line_{i}" for i in range(1000)]),
        "should_score_max": 0.4,  # Should be penalized for size
    },
    {
        "name": "syntax_error",
        "description": "Patch with Python syntax errors",
        "patch": """--- a/example.py
+++ b/example.py
@@ -10,7 +10,7 @@
 def foo():
-    return "old"
+    return "new"
+    def broken_syntax(
""",
        "should_score_max": 0.3,  # Should fail to apply or score low
    },
]


class GraderCalibrator:
    def __init__(self, grader_url: str = GRADER_ENDPOINT):
        self.grader_url = grader_url
        self.results = {
            "hackability_tests": [],
            "baseline_scores": [],
            "threshold_calibration": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    def check_grader_health(self) -> bool:
        """Check if grader endpoint is available."""
        try:
            response = requests.get(f"{self.grader_url}/health", timeout=5)
            response.raise_for_status()
            print(f"✓ Grader healthy: {response.json()}")
            return True
        except Exception as e:
            print(f"✗ Grader not available: {e}")
            return False
    
    def run_hackability_tests(self, instance_id: str, repo: str, base_commit: str) -> bool:
        """
        Test if grader can be gamed with bad patches.
        Returns True if all hack attempts are properly caught.
        """
        print("\n" + "=" * 60)
        print("HACKABILITY TESTS")
        print("=" * 60)
        
        all_passed = True
        
        for test in HACK_TEST_PATCHES:
            print(f"\nTest: {test['name']}")
            print(f"  Description: {test['description']}")
            
            try:
                response = requests.post(
                    f"{self.grader_url}/grade",
                    json={
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": base_commit,
                        "patch": test["patch"],
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                
                score = result.get("score", 0)
                error = result.get("error")
                
                # Check if hack was caught
                if "should_score" in test:
                    passed = score == test["should_score"]
                elif "should_score_max" in test:
                    passed = score <= test["should_score_max"]
                else:
                    passed = True
                
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  Score: {score:.4f} | {status}")
                if error:
                    print(f"  Error: {error}")
                
                if not passed:
                    all_passed = False
                    print(f"  WARNING: Hack test '{test['name']}' scored too high!")
                
                self.results["hackability_tests"].append({
                    "name": test["name"],
                    "score": score,
                    "passed": passed,
                    "error": error,
                })
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_passed = False
        
        return all_passed
    
    def calibrate_threshold(
        self,
        instances: list[dict],
        target_failure_rate: float = 0.375,  # 25-50% failure
    ) -> float:
        """
        Calibrate pass_threshold so base model fails 25-50% of instances.
        Returns optimal threshold.
        """
        print("\n" + "=" * 60)
        print("THRESHOLD CALIBRATION")
        print("=" * 60)
        
        scores = []
        
        for inst in instances:
            print(f"\nInstance: {inst['instance_id']}")
            
            try:
                response = requests.post(
                    f"{self.grader_url}/grade",
                    json={
                        "instance_id": inst["instance_id"],
                        "repo": inst["repo"],
                        "base_commit": inst["base_commit"],
                        "patch": inst.get("patch", "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-placeholder"),
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                
                score = result.get("score", 0)
                scores.append(score)
                print(f"  Score: {score:.4f}")
                
                self.results["baseline_scores"].append({
                    "instance_id": inst["instance_id"],
                    "score": score,
                })
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        if not scores:
            print("No scores collected!")
            return 0.5
        
        scores_array = np.array(scores)
        
        # Find threshold where target_failure_rate of instances fail
        # "Fail" = score < threshold
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_diff = float("inf")
        
        for t in thresholds:
            failure_rate = np.mean(scores_array < t)
            diff = abs(failure_rate - target_failure_rate)
            if diff < best_diff:
                best_diff = diff
                best_threshold = t
        
        actual_failure_rate = np.mean(scores_array < best_threshold)
        
        print(f"\nCalibration results:")
        print(f"  Scores: min={scores_array.min():.4f}, max={scores_array.max():.4f}, "
              f"mean={scores_array.mean():.4f}, std={scores_array.std():.4f}")
        print(f"  Optimal threshold: {best_threshold:.4f}")
        print(f"  Target failure rate: {target_failure_rate:.2%}")
        print(f"  Actual failure rate: {actual_failure_rate:.2%}")
        
        self.results["threshold_calibration"] = {
            "optimal_threshold": float(best_threshold),
            "target_failure_rate": target_failure_rate,
            "actual_failure_rate": float(actual_failure_rate),
            "scores_stats": {
                "min": float(scores_array.min()),
                "max": float(scores_array.max()),
                "mean": float(scores_array.mean()),
                "std": float(scores_array.std()),
            },
        }
        
        return best_threshold
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save calibration results."""
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / "calibration_results.json"
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """Print calibration summary."""
        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)
        
        # Hackability
        hack_tests = self.results["hackability_tests"]
        hack_passed = sum(1 for t in hack_tests if t["passed"])
        print(f"\nHackability: {hack_passed}/{len(hack_tests)} tests passed")
        
        if hack_passed < len(hack_tests):
            print("⚠ WARNING: Some hack tests failed. Grader may be vulnerable!")
            print("  → REDISEÑAR GRADER BEFORE TRAINING")
        else:
            print("✓ All hack tests passed")
        
        # Threshold
        cal = self.results["threshold_calibration"]
        if cal:
            print(f"\nThreshold: {cal['optimal_threshold']:.4f}")
            print(f"Failure rate: {cal['actual_failure_rate']:.2%} "
                  f"(target: {cal['target_failure_rate']:.2%})")
        
        # Recommendation
        print("\n" + "-" * 60)
        if hack_passed < len(hack_tests):
            print("⛔ RECOMMENDATION: DO NOT PROCEED WITH TRAINING")
            print("   Grader is vulnerable to reward hacking.")
        elif cal and abs(cal['actual_failure_rate'] - cal['target_failure_rate']) > 0.1:
            print("⚠ RECOMMENDATION: RECALIBRATE THRESHOLD")
            print("   Failure rate deviates >10% from target.")
        else:
            print("✓ RECOMMENDATION: PROCEED WITH TRAINING")
        print("-" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate SWE-bench grader")
    parser.add_argument("--grader-url", default=GRADER_ENDPOINT, help="Grader endpoint URL")
    parser.add_argument("--instances-file", help="JSON file with instances for calibration")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    calibrator = GraderCalibrator(args.grader_url)
    
    # Check grader health
    if not calibrator.check_grader_health():
        print("Grader not available. Start it first:")
        print("  python scripts/grader_endpoint.py")
        sys.exit(1)
    
    # Load instances
    instances = []
    if args.instances_file:
        with open(args.instances_file) as f:
            instances = json.load(f)
    
    # Default test instances (use a small subset of SWE-bench)
    if not instances:
        print("No instances file provided. Using default test instances.")
        print("For full calibration, provide --instances-file with SWE-bench instances.")
        instances = [
            {
                "instance_id": "django__django-11099",
                "repo": "django/django",
                "base_commit": "0454282c39925f56fc478c0632077ab05b79a14f",
            },
            {
                "instance_id": "sympy__sympy-18057",
                "repo": "sympy/sympy",
                "base_commit": "b3ea5c42c3463e2a2c7995323cf00eff6c213cb5",
            },
            {
                "instance_id": "requests__requests-3382",
                "repo": "psf/requests",
                "base_commit": "e41ce45976852b8e8d5506a3e3e4f7e0f40e3e5e",
            },
        ]
    
    # Run calibration
    hackability_ok = calibrator.run_hackability_tests(
        instance_id=instances[0]["instance_id"],
        repo=instances[0]["repo"],
        base_commit=instances[0]["base_commit"],
    )
    
    threshold = calibrator.calibrate_threshold(instances)
    
    # Save results
    calibrator.save_results()
    calibrator.print_summary()
    
    # Exit with error if calibration failed
    if not hackability_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
