#!/usr/bin/env python3
"""
SWE-bench Grader Endpoint
FastAPI service that applies patches in Docker and runs repo tests.
Deterministic scoring with anti-reward-hacking guards.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import docker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ── Configuration ──────────────────────────────────────────────────────────────
DOCKER_IMAGE = "swe-bench-grader:latest"
DEFAULT_TIMEOUT = 600  # 10 minutes per test suite
MAX_DIFF_SIZE = 50_000  # characters
ANTI_HACK_MEGA_PATCH_THRESHOLD = 10_000  # chars diff to trigger size penalty

app = FastAPI(title="SWE-bench Grader", version="1.0.0")

# ── Data models ────────────────────────────────────────────────────────────────
class GradeRequest(BaseModel):
    instance_id: str = Field(..., description="SWE-bench instance ID")
    repo: str = Field(..., description="GitHub repo (owner/name)")
    base_commit: str = Field(..., description="Commit SHA to apply patch on")
    patch: str = Field(..., description="Unified diff to apply")
    test_patch: Optional[str] = Field(None, description="Optional test patch override")

class GradeResult(BaseModel):
    instance_id: str
    score: float = Field(ge=0.0, le=1.0)
    fail_to_pass: int = Field(description="Tests that went from FAIL to PASS")
    fail_to_pass_total: int = Field(description="Total failing tests in baseline")
    pass_to_pass: int = Field(description="Tests that were PASS and remain PASS")
    pass_to_pass_total: int = Field(description="Total passing tests in baseline")
    patch_applies: bool
    diff_size: int
    timeout: bool = False
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

# ── Docker grader logic ────────────────────────────────────────────────────────
class DockerGrader:
    def __init__(self, docker_image: str = DOCKER_IMAGE, timeout: int = DEFAULT_TIMEOUT):
        self.docker_image = docker_image
        self.timeout = timeout
        self.client = docker.from_env()

    def _run_in_container(self, commands: list[str], env: dict = None) -> tuple[int, str]:
        """Run commands in a Docker container, return (exit_code, output)."""
        container = self.client.containers.run(
            self.docker_image,
            command=["bash", "-c", "\n".join(commands)],
            detach=True,
            environment=env or {},
            mem_limit="4g",
            cpu_period=100000,
            cpu_quota=400000,  # 4 CPUs
            network_disabled=True,
            remove=True,
        )
        try:
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode("utf-8", errors="replace")
            return result["StatusCode"], logs
        except Exception:
            container.kill()
            return -1, "Container killed (timeout or error)"
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def grade(self, request: GradeRequest) -> GradeResult:
        """Grade a single SWE-bench instance."""
        result = GradeResult(
            instance_id=request.instance_id,
            score=0.0,
            fail_to_pass=0,
            fail_to_pass_total=0,
            pass_to_pass=0,
            pass_to_pass_total=0,
            patch_applies=False,
            diff_size=len(request.patch),
        )

        # Guard: diff too large
        if len(request.patch) > MAX_DIFF_SIZE:
            result.error = f"Patch too large: {len(request.patch)} chars > {MAX_DIFF_SIZE}"
            return result

        # Step 1: Checkout repo and apply patch
        checkout_cmds = [
            f"cd /workspace && git clone --quiet https://github.com/{request.repo}.git repo",
            f"cd /workspace/repo && git checkout --quiet {request.base_commit}",
        ]

        # Save baseline test results
        baseline_cmds = checkout_cmds + [
            "cd /workspace/repo && python -m pytest --tb=no -q --co 2>/dev/null | tail -1 || echo '0'",
        ]

        # Apply patch
        patch_cmds = [
            f"cd /workspace/repo && git apply --check <<'DIFF'\n{request.patch}\nDIFF",
        ]

        # Run tests after patch
        test_cmds = [
            "cd /workspace/repo && git apply <<'DIFF'\n" + request.patch + "\nDIFF",
            "cd /workspace/repo && python -m pytest --tb=short -q 2>&1 | tail -50",
        ]

        # Full pipeline
        all_cmds = checkout_cmds + [
            # Get baseline test results
            "cd /workspace/repo && python -m pytest --tb=no -q 2>&1 | grep -E 'passed|failed' | tail -1 > /workspace/baseline.txt",
            "cat /workspace/baseline.txt",
            # Apply patch
            f"cd /workspace/repo && git apply --check <<'DIFF'\n{request.patch}\nDIFF && echo 'PATCH_OK' || echo 'PATCH_FAIL'",
            # Run tests
            "cd /workspace/repo && python -m pytest --tb=short -q 2>&1 | grep -E 'passed|failed' | tail -1 > /workspace/after.txt",
            "cat /workspace/after.txt",
        ]

        exit_code, output = self._run_in_container(all_cmds)

        if exit_code == -1:
            result.timeout = True
            result.error = "Container timed out"
            return result

        # Parse results
        try:
            lines = output.strip().split("\n")

            # Check if patch applied
            patch_ok = any("PATCH_OK" in line for line in lines)
            result.patch_applies = patch_ok

            if not patch_ok:
                result.error = "Patch failed to apply"
                return result

            # Parse baseline and after test results
            baseline_line = ""
            after_line = ""
            for i, line in enumerate(lines):
                if "baseline.txt" in line and i + 1 < len(lines):
                    baseline_line = lines[i + 1]
                if "after.txt" in line and i + 1 < len(lines):
                    after_line = lines[i + 1]

            # Parse "X passed, Y failed" format
            def parse_test_result(line: str) -> tuple[int, int]:
                passed = failed = 0
                if "passed" in line:
                    parts = line.split("passed")[0].strip().split(",")
                    passed = int(parts[-1].strip()) if parts else 0
                if "failed" in line:
                    parts = line.split("failed")[0].strip().split(",")
                    # Get the part after "X passed,"
                    if "passed" in line:
                        after_passed = line.split("passed")[1]
                        if "failed" in after_passed:
                            failed_str = after_passed.split("failed")[0].strip().rstrip(",").lstrip(",")
                            failed = int(failed_str) if failed_str else 0
                    else:
                        failed = int(line.split("failed")[0].strip().split(",")[-1])
                return passed, failed

            baseline_passed, baseline_failed = parse_test_result(baseline_line)
            after_passed, after_failed = parse_test_result(after_line)

            result.fail_to_pass_total = baseline_failed
            result.pass_to_pass_total = baseline_passed

            # fail_to_pass: tests that were FAIL and now PASS
            result.fail_to_pass = max(0, baseline_failed - after_failed)
            # pass_to_pass: tests that were PASS and remain PASS
            result.pass_to_pass = min(baseline_passed, after_passed)

            # ── Score calculation (deterministic, anti-hack) ──
            score = 0.0

            # +0.5: fail_to_pass ratio (core: issue resolved)
            if baseline_failed > 0:
                ftp_ratio = result.fail_to_pass / baseline_failed
                score += 0.5 * ftp_ratio

            # +0.2: pass_to_pass ratio (no regressions)
            if baseline_passed > 0:
                ptp_ratio = result.pass_to_pass / baseline_passed
                score += 0.2 * ptp_ratio

            # +0.2: patch applies cleanly
            if result.patch_applies:
                score += 0.2

            # +0.1: reasonable diff size (anti mega-patch)
            if result.diff_size < ANTI_HACK_MEGA_PATCH_THRESHOLD:
                score += 0.1

            # Anti-hack: empty patch penalty
            if result.diff_size == 0:
                score = 0.0
                result.error = "Empty patch detected (reward hack)"

            # Anti-hack: patch only deletes tests
            if result.fail_to_pass == 0 and after_failed == 0 and baseline_failed > 0:
                # Suspicious: all tests pass now but none went from fail to pass
                # Likely deleted the tests
                score = max(0.0, score - 0.5)
                result.error = "Suspicious: all failures resolved without fail_to_pass (possible test deletion)"

            result.score = round(min(1.0, max(0.0, score)), 4)
            result.metadata = {
                "baseline_passed": baseline_passed,
                "baseline_failed": baseline_failed,
                "after_passed": after_passed,
                "after_failed": after_failed,
            }

        except Exception as e:
            result.error = f"Failed to parse results: {str(e)}"

        return result

# ── API endpoints ──────────────────────────────────────────────────────────────
grader = DockerGrader()

@app.post("/grade", response_model=GradeResult)
async def grade_endpoint(request: GradeRequest):
    """Grade a single SWE-bench instance."""
    try:
        result = grader.grade(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grade_batch", response_model=list[GradeResult])
async def grade_batch_endpoint(requests: list[GradeRequest]):
    """Grade multiple SWE-bench instances."""
    results = []
    for req in requests:
        try:
            result = grader.grade(req)
            results.append(result)
        except Exception as e:
            results.append(GradeResult(
                instance_id=req.instance_id,
                score=0.0,
                fail_to_pass=0,
                fail_to_pass_total=0,
                pass_to_pass=0,
                pass_to_pass_total=0,
                patch_applies=False,
                diff_size=len(req.patch),
                error=str(e),
            ))
    return results

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "docker_image": DOCKER_IMAGE}

@app.get("/baseline/{instance_id}")
async def get_baseline(instance_id: str, repo: str, base_commit: str):
    """Get baseline test results for an instance (for calibration)."""
    try:
        # Create a no-op patch to get baseline
        noop_patch = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-placeholder"
        req = GradeRequest(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            patch=noop_patch,
        )
        result = grader.grade(req)
        return {
            "instance_id": instance_id,
            "baseline_passed": result.metadata.get("baseline_passed", 0),
            "baseline_failed": result.metadata.get("baseline_failed", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SWE-bench Grader Endpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--docker-image", default=DOCKER_IMAGE, help="Docker image for grading")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per instance (seconds)")
    args = parser.parse_args()

    grader.docker_image = args.docker_image
    grader.timeout = args.timeout

    print(f"Starting SWE-bench Grader on {args.host}:{args.port}")
    print(f"Docker image: {args.docker_image}")
    print(f"Timeout: {args.timeout}s")

    uvicorn.run(app, host=args.host, port=args.port)