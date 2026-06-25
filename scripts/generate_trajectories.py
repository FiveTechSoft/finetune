#!/usr/bin/env python3
"""
Trajectory Generator for SFT
Distills trajectories from a strong coder agent (Qwen-Coder with SWE-agent scaffolding).
Filters to only include trajectories that pass tests.
"""

import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import requests
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
GRADER_ENDPOINT = "http://localhost:8000"
BASE_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
SWE_BENCH_TRAIN_PATH = Path("/home/fivetech/finetune/data/swe_bench_train.json")
OUTPUT_DIR = Path("/home/fivetech/finetune/data/sft_trajectories")
MAX_TRAJECTORIES = 1000
MIN_QUALITY_SCORE = 0.5  # Minimum grader score to keep trajectory

# ── Data models ────────────────────────────────────────────────────────────────
@dataclass
class Trajectory:
    instance_id: str
    repo: str
    base_commit: str
    messages: list[dict]
    patch: str
    grader_score: float
    quality_score: float
    metadata: dict


class TrajectoryGenerator:
    def __init__(
        self,
        grader_url: str = GRADER_ENDPOINT,
        model_name: str = BASE_MODEL,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.grader_url = grader_url
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trajectories = []
        self.stats = {
            "total_generated": 0,
            "passed_grading": 0,
            "deduped": 0,
            "quality_filtered": 0,
        }
    
    def load_swe_bench_instances(self, path: Path) -> list[dict]:
        """Load SWE-bench training instances."""
        if not path.exists():
            print(f"Warning: {path} not found. Using sample instances.")
            return self._get_sample_instances()
        
        with open(path) as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} SWE-bench instances from {path}")
        return data
    
    def _get_sample_instances(self) -> list[dict]:
        """Sample SWE-bench instances for testing."""
        return [
            {
                "instance_id": "django__django-11099",
                "repo": "django/django",
                "base_commit": "0454282c39925f56fc478c0632077ab05b79a14f",
                "problem_statement": "Fix URL validation to allow underscores...",
                "hints_text": "The regex pattern needs to be updated...",
                "created_at": "2019-06-25",
            },
            {
                "instance_id": "sympy__sympy-18057",
                "repo": "sympy/sympy",
                "base_commit": "b3ea5c42c3463e2a2c7995323cf00eff6c213cb5",
                "problem_statement": "Fix simplify() for matrix expressions...",
                "hints_text": "The _eval_simplify method needs...",
                "created_at": "2020-03-19",
            },
            {
                "instance_id": "requests__requests-3382",
                "repo": "psf/requests",
                "base_commit": "e41ce45976852b8e8d5506a3e3e4f7e0f40e3e5e",
                "problem_statement": "Fix redirect handling with cookies...",
                "hints_text": "The Session.resolve_redirects method...",
                "created_at": "2020-01-15",
            },
        ]
    
    def generate_trajectory(
        self,
        instance: dict,
        max_retries: int = 3,
    ) -> Optional[Trajectory]:
        """
        Generate a trajectory for a single SWE-bench instance.
        Uses the model to generate a patch, then validates with grader.
        """
        instance_id = instance["instance_id"]
        
        # Construct prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineer. Analyze the issue, "
                    "explore the repository, and provide a fix as a unified diff.\n"
                    "Format your response as:\n"
                    "1. Analysis of the issue\n"
                    "2. Files to modify\n"
                    "<patch>\nunified diff here\n</patch>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Repository: {instance['repo']}\n"
                    f"Commit: {instance['base_commit']}\n\n"
                    f"Issue:\n{instance.get('problem_statement', 'No description')}\n\n"
                    f"Hints:\n{instance.get('hints_text', 'No hints')}"
                ),
            },
        ]
        
        # Generate completion (simulated - in production, call model API)
        # For now, we'll create a placeholder trajectory
        # In production, this would call the actual model
        
        patch = self._generate_patch_placeholder(instance)
        
        if not patch:
            return None
        
        # Validate with grader
        try:
            response = requests.post(
                f"{self.grader_url}/grade",
                json={
                    "instance_id": instance_id,
                    "repo": instance["repo"],
                    "base_commit": instance["base_commit"],
                    "patch": patch,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            
            score = result.get("score", 0)
            
            # Only keep trajectories that pass tests
            if score < MIN_QUALITY_SCORE:
                return None
            
            # Add completion to messages
            messages.append({
                "role": "assistant",
                "content": (
                    f"I've analyzed the issue and created a fix.\n\n"
                    f"<patch>\n{patch}\n</patch>"
                ),
            })
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(result)
            
            trajectory = Trajectory(
                instance_id=instance_id,
                repo=instance["repo"],
                base_commit=instance["base_commit"],
                messages=messages,
                patch=patch,
                grader_score=score,
                quality_score=quality_score,
                metadata={
                    "model": self.model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "grader_result": result,
                },
            )
            
            return trajectory
            
        except Exception as e:
            print(f"  Grader error for {instance_id}: {e}")
            return None
    
    def _generate_patch_placeholder(self, instance: dict) -> Optional[str]:
        """
        Placeholder patch generation.
        In production, this would call the actual model API.
        """
        # This is a simplified placeholder
        # In production, you would:
        # 1. Clone the repo
        # 2. Checkout the base commit
        # 3. Call the model with the problem statement
        # 4. Extract the patch from the response
        
        repo = instance["repo"]
        
        # Placeholder patch for testing
        patch = f"""--- a/README.md
+++ b/README.md
@@ -1,5 +1,6 @@
 # {repo.split('/')[-1]}
 
+Fixed: {instance.get('problem_statement', 'issue')[:50]}
+
 ## Description
 
 A project description.
"""
        return patch
    
    def _calculate_quality_score(self, grader_result: dict) -> float:
        """Calculate quality score from grader result."""
        score = grader_result.get("score", 0)
        
        # Bonus for clean patch
        if grader_result.get("patch_applies"):
            score += 0.1
        
        # Penalty for large diff
        diff_size = grader_result.get("diff_size", 0)
        if diff_size > 5000:
            score -= 0.1
        
        return round(min(1.0, max(0.0, score)), 4)
    
    def deduplicate(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Remove duplicate trajectories (same instance_id, keep highest quality)."""
        seen = {}
        
        for traj in trajectories:
            key = traj.instance_id
            if key not in seen or traj.quality_score > seen[key].quality_score:
                seen[key] = traj
        
        deduped = list(seen.values())
        self.stats["deduped"] = len(trajectories) - len(deduped)
        
        return deduped
    
    def format_for_sft(self, trajectories: list[Trajectory]) -> list[dict]:
        """Format trajectories for SFT training."""
        formatted = []
        
        for traj in trajectories:
            formatted.append({
                "messages": traj.messages,
                "metadata": {
                    "instance_id": traj.instance_id,
                    "repo": traj.repo,
                    "grader_score": traj.grader_score,
                    "quality_score": traj.quality_score,
                },
            })
        
        return formatted
    
    def save_trajectories(self, trajectories: list[Trajectory], filename: str = "sft_trajectories.jsonl"):
        """Save trajectories to JSONL file."""
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            for traj in trajectories:
                f.write(json.dumps(asdict(traj)) + "\n")
        
        print(f"\nSaved {len(trajectories)} trajectories to {output_path}")
        return output_path
    
    def generate(
        self,
        instances: list[dict],
        max_trajectories: int = MAX_TRAJECTORIES,
    ) -> list[Trajectory]:
        """Generate trajectories for all instances."""
        print("\n" + "=" * 60)
        print("TRAJECTORY GENERATION")
        print("=" * 60)
        
        all_trajectories = []
        
        for i, instance in enumerate(tqdm(instances[:max_trajectories], desc="Generating")):
            self.stats["total_generated"] += 1
            
            trajectory = self.generate_trajectory(instance)
            
            if trajectory:
                all_trajectories.append(trajectory)
                self.stats["passed_grading"] += 1
            
            # Rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(1)
        
        # Deduplicate
        deduped = self.deduplicate(all_trajectories)
        
        # Filter by quality
        quality_filtered = [
            t for t in deduped
            if t.quality_score >= MIN_QUALITY_SCORE
        ]
        self.stats["quality_filtered"] = len(deduped) - len(quality_filtered)
        
        # Format for SFT
        formatted = self.format_for_sft(quality_filtered)
        
        # Save
        self.save_trajectories(quality_filtered)
        
        # Save formatted version
        formatted_path = self.output_dir / "sft_formatted.jsonl"
        with open(formatted_path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")
        
        print(f"\nFormatted SFT data saved to {formatted_path}")
        
        # Print stats
        print("\n" + "=" * 60)
        print("GENERATION STATS")
        print("=" * 60)
        print(f"Total instances: {self.stats['total_generated']}")
        print(f"Passed grading: {self.stats['passed_grading']}")
        print(f"Deduped: {self.stats['deduped']}")
        print(f"Quality filtered: {self.stats['quality_filtered']}")
        print(f"Final trajectories: {len(quality_filtered)}")
        
        return quality_filtered


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    parser.add_argument("--grader-url", default=GRADER_ENDPOINT, help="Grader endpoint URL")
    parser.add_argument("--model", default=BASE_MODEL, help="Model name")
    parser.add_argument("--instances-file", type=Path, help="SWE-bench instances file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-trajectories", type=int, default=MAX_TRAJECTORIES, help="Max trajectories")
    args = parser.parse_args()
    
    generator = TrajectoryGenerator(
        grader_url=args.grader_url,
        model_name=args.model,
        output_dir=args.output_dir,
    )
    
    # Load instances
    instances = generator.load_swe_bench_instances(args.instances_file)
    
    # Generate trajectories
    trajectories = generator.generate(instances, args.max_trajectories)
    
    print(f"\nDone! Generated {len(trajectories)} trajectories.")


if __name__ == "__main__":
    main()
