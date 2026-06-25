#!/usr/bin/env python3
"""
GRPO Reward Functions for SWE-bench
Reward functions that call the grader endpoint to score model patches.
"""

import json
import re
import requests
from typing import Optional
from dataclasses import dataclass

# ── Configuration ──────────────────────────────────────────────────────────────
GRADER_ENDPOINT = "http://localhost:8000"
DEFAULT_TIMEOUT = 600  # 10 minutes

# ── Reward function data ───────────────────────────────────────────────────────
@dataclass
class RewardResult:
    score: float
    components: dict
    metadata: dict
    error: Optional[str] = None

# ── Core reward functions ──────────────────────────────────────────────────────
def swe_bench_reward(
    instance_id: str,
    repo: str,
    base_commit: str,
    patch: str,
    grader_url: str = GRADER_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT,
) -> RewardResult:
    """
    Main reward function for SWE-bench GRPO.
    Calls the grader endpoint to apply patch and run tests.
    
    Score components (total 1.0):
    - 0.5: fail_to_pass ratio (issue resolved)
    - 0.2: pass_to_pass ratio (no regressions)
    - 0.2: patch applies cleanly
    - 0.1: reasonable diff size
    """
    components = {
        "fail_to_pass": 0.0,
        "pass_to_pass": 0.0,
        "patch_applies": 0.0,
        "diff_size": 0.0,
    }
    
    # Validate patch format
    if not patch or not patch.strip():
        return RewardResult(
            score=0.0,
            components=components,
            metadata={"error": "empty_patch"},
            error="Empty patch"
        )
    
    # Check for unified diff format
    if not re.search(r"^@@.*@@", patch, re.MULTILINE):
        return RewardResult(
            score=0.0,
            components=components,
            metadata={"error": "invalid_diff_format"},
            error="Invalid diff format"
        )
    
    # Call grader endpoint
    try:
        response = requests.post(
            f"{grader_url}/grade",
            json={
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit,
                "patch": patch,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.Timeout:
        return RewardResult(
            score=0.0,
            components=components,
            metadata={"error": "timeout"},
            error="Grader timeout"
        )
    except requests.exceptions.RequestException as e:
        return RewardResult(
            score=0.0,
            components=components,
            metadata={"error": "grader_unavailable"},
            error=f"Grader unavailable: {str(e)}"
        )
    
    # Calculate component scores
    score = 0.0
    
    # +0.5: fail_to_pass ratio
    if result.get("fail_to_pass_total", 0) > 0:
        ftp_ratio = result["fail_to_pass"] / result["fail_to_pass_total"]
        components["fail_to_pass"] = 0.5 * ftp_ratio
        score += components["fail_to_pass"]
    
    # +0.2: pass_to_pass ratio
    if result.get("pass_to_pass_total", 0) > 0:
        ptp_ratio = result["pass_to_pass"] / result["pass_to_pass_total"]
        components["pass_to_pass"] = 0.2 * ptp_ratio
        score += components["pass_to_pass"]
    
    # +0.2: patch applies cleanly
    if result.get("patch_applies", False):
        components["patch_applies"] = 0.2
        score += 0.2
    
    # +0.1: reasonable diff size
    diff_size = result.get("diff_size", 0)
    if diff_size < 10_000:
        components["diff_size"] = 0.1
        score += 0.1
    
    # Anti-hack penalties
    if result.get("error") == "Empty patch detected (reward hack)":
        score = 0.0
    elif result.get("error") and "Suspicious" in result.get("error", ""):
        score = max(0.0, score - 0.5)
    
    return RewardResult(
        score=round(min(1.0, max(0.0, score)), 4),
        components=components,
        metadata={
            "fail_to_pass": result.get("fail_to_pass", 0),
            "fail_to_pass_total": result.get("fail_to_pass_total", 0),
            "pass_to_pass": result.get("pass_to_pass", 0),
            "pass_to_pass_total": result.get("pass_to_pass_total", 0),
            "patch_applies": result.get("patch_applies", False),
            "diff_size": diff_size,
        },
        error=result.get("error"),
    )


def format_reward(response_text: str) -> RewardResult:
    """
    Auxiliary reward for proper tool-use format.
    Rewards responses that follow the expected format:
    - Contains <patch> tags with unified diff
    - Proper structure
    """
    score = 0.0
    components = {
        "has_patch_tag": 0.0,
        "valid_diff": 0.0,
        "no_excessive_text": 0.0,
    }
    
    # Check for <patch> tags
    if "<patch>" in response_text and "</patch>" in response_text:
        components["has_patch_tag"] = 0.4
        score += 0.4
        
        # Extract and validate diff
        patch_match = re.search(r"<patch>(.*?)</patch>", response_text, re.DOTALL)
        if patch_match:
            patch = patch_match.group(1).strip()
            if re.search(r"^@@.*@@", patch, re.MULTILINE):
                components["valid_diff"] = 0.4
                score += 0.4
    
    # Penalize excessive text (keep responses concise)
    text_length = len(response_text)
    if text_length < 2000:
        components["no_excessive_text"] = 0.2
        score += 0.2
    elif text_length < 5000:
        components["no_excessive_text"] = 0.1
        score += 0.1
    
    return RewardResult(
        score=round(min(1.0, max(0.0, score)), 4),
        components=components,
        metadata={"text_length": text_length},
    )


def combined_reward(
    instance_id: str,
    repo: str,
    base_commit: str,
    patch: str,
    response_text: str,
    grader_url: str = GRADER_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT,
    swe_weight: float = 0.8,
    format_weight: float = 0.2,
) -> RewardResult:
    """
    Combined reward: SWE-bench (test execution) + format auxiliary.
    Default weights: 80% test execution, 20% format.
    """
    swe_result = swe_bench_reward(
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        patch=patch,
        grader_url=grader_url,
        timeout=timeout,
    )
    
    format_result = format_reward(response_text)
    
    combined_score = (
        swe_weight * swe_result.score +
        format_weight * format_result.score
    )
    
    return RewardResult(
        score=round(min(1.0, max(0.0, combined_score)), 4),
        components={
            "swe_bench": swe_result.components,
            "format": format_result.components,
            "weights": {"swe": swe_weight, "format": format_weight},
        },
        metadata={
            "swe_score": swe_result.score,
            "format_score": format_result.score,
            "swe_metadata": swe_result.metadata,
        },
        error=swe_result.error or format_result.error,
    )


# ── GRPO-compatible reward wrapper ─────────────────────────────────────────────
def grpo_reward_fn(
    prompts: list[str],
    completions: list[str],
    instance_ids: list[str],
    repos: list[str],
    base_commits: list[str],
    grader_url: str = GRADER_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[float]:
    """
    GRPO-compatible reward function.
    Takes batch of prompts and completions, returns list of scores.
    """
    scores = []
    
    for prompt, completion, instance_id, repo, base_commit in zip(
        prompts, completions, instance_ids, repos, base_commits
    ):
        # Extract patch from completion
        patch_match = re.search(r"<patch>(.*?)</patch>", completion, re.DOTALL)
        patch = patch_match.group(1).strip() if patch_match else ""
        
        result = combined_reward(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            patch=patch,
            response_text=completion,
            grader_url=grader_url,
            timeout=timeout,
        )
        
        scores.append(result.score)
    
    return scores


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test reward functions")
    parser.add_argument("--grader-url", default=GRADER_ENDPOINT, help="Grader endpoint URL")
    args = parser.parse_args()
    
    # Test format reward
    test_response = """
    I'll analyze the issue and create a patch.
    
    <patch>
--- a/example.py
+++ b/example.py
@@ -10,7 +10,7 @@
 def foo():
-    return "old"
+    return "new"
</patch>
    """
    
    result = format_reward(test_response)
    print(f"Format reward: {result.score}")
    print(f"Components: {result.components}")
    
    # Test connection to grader
    try:
        response = requests.get(f"{args.grader_url}/health", timeout=5)
        print(f"\nGrader health: {response.json()}")
    except Exception as e:
        print(f"\nGrader not available: {e}")
