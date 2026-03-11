"""Run insurance-rdd tests on Databricks — writes output to Workspace for retrieval."""

import os
import sys
import time
import base64
import glob

# Load env.
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service import jobs as jobs_service

w = WorkspaceClient()

PROJECT = "/home/ralph/insurance-rdd"
WORKSPACE_PATH = "/Workspace/insurance-rdd-tests"
RESULTS_PATH = f"{WORKSPACE_PATH}/test_results.txt"

print("Uploading project files to Databricks workspace...")


def upload_file(local_path: str, remote_path: str) -> None:
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    try:
        w.workspace.delete(path=remote_path, recursive=False)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        format=ImportFormat.AUTO,
        overwrite=True,
        content=content,
    )


# Create workspace directories.
for path in [
    WORKSPACE_PATH,
    f"{WORKSPACE_PATH}/src",
    f"{WORKSPACE_PATH}/src/insurance_rdd",
    f"{WORKSPACE_PATH}/tests",
]:
    try:
        w.workspace.mkdirs(path=path)
    except Exception:
        pass

# Upload source files.
for src_file in glob.glob(f"{PROJECT}/src/insurance_rdd/*.py"):
    fname = os.path.basename(src_file)
    remote = f"{WORKSPACE_PATH}/src/insurance_rdd/{fname}"
    upload_file(src_file, remote)
    print(f"  Uploaded src: {fname}")

# Upload test files.
for test_file in glob.glob(f"{PROJECT}/tests/*.py"):
    fname = os.path.basename(test_file)
    remote = f"{WORKSPACE_PATH}/tests/{fname}"
    upload_file(test_file, remote)
    print(f"  Uploaded test: {fname}")

print("\nCreating test runner notebook...")

# The notebook writes test output to a Workspace file via base64 encoding.
# This avoids the DBFS public root which is disabled on this workspace.
notebook_content = '''# Databricks notebook source
# MAGIC %pip install "numpy==1.24.4" "scipy==1.10.1" rdrobust rddensity rdlocrand rdmulti matplotlib pytest --quiet

# COMMAND ----------

import sys, os, shutil, glob, subprocess, base64
from scipy import optimize, stats
import numpy as np
print(f"numpy: {np.__version__}, scipy: {__import__('scipy').__version__}")

src_dir = "/tmp/insurance_rdd_src"
test_dir = "/tmp/insurance_rdd_tests"
os.makedirs(f"{src_dir}/insurance_rdd", exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for f in glob.glob("/Workspace/insurance-rdd-tests/src/insurance_rdd/*.py"):
    shutil.copy(f, f"{src_dir}/insurance_rdd/{os.path.basename(f)}")
for f in glob.glob("/Workspace/insurance-rdd-tests/tests/*.py"):
    shutil.copy(f, f"{test_dir}/{os.path.basename(f)}")

if src_dir not in sys.path: sys.path.insert(0, src_dir)
if test_dir not in sys.path: sys.path.insert(0, test_dir)

import insurance_rdd
print(f"insurance-rdd {insurance_rdd.__version__} imported OK")

# COMMAND ----------

result = subprocess.run(
    [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short",
     "--ignore=" + os.path.join(test_dir, "test_plots.py"),
     "-p", "no:warnings", "--no-header"],
    capture_output=True, text=True, cwd=test_dir,
    env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": src_dir},
)
full_out = result.stdout
if result.stderr:
    full_out += "\\n=== STDERR ===\\n" + result.stderr

exit_code = result.returncode
full_out += f"\\n\\nFINAL_EXIT_CODE: {exit_code}"

# Write output to Workspace file using the workspace API directly.
# Encode as base64 and import as a raw file.
results_b64 = base64.b64encode(full_out.encode("utf-8")).decode("utf-8")

# Use dbutils REST API to write to workspace.
# We'll use a simpler approach: write to /tmp and use the notebook exit value
# to pass the encoded content. But that's limited to 1MB.
# Instead, write to /Workspace via subprocess calling Databricks CLI or
# by importing via the workspace REST API using requests.

import requests, json

# Get the token from the environment (set by Databricks cluster).
token = os.environ.get("DATABRICKS_TOKEN", "")
host = os.environ.get("DATABRICKS_HOST", "")

# Try to get from spark config if env vars not set.
try:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    print(f"Got credentials from dbutils: host={host}")
except Exception as e:
    print(f"Could not get credentials from dbutils: {e}")

if token and host:
    resp = requests.post(
        f"{host}/api/2.0/workspace/import",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "path": "/Workspace/insurance-rdd-tests/test_results.txt",
            "format": "AUTO",
            "overwrite": True,
            "content": results_b64,
        },
    )
    print(f"Workspace import status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Response: {resp.text}")
        # Fall back: print first 8000 chars of output so we get something.
        print("=== FALLBACK: printing output directly ===")
        print(full_out[:8000])
else:
    print("No credentials — printing output directly")
    print(full_out[:8000])

# Print last bit for direct log visibility.
print("\\n=== TEST OUTPUT (last 3000 chars) ===")
print(full_out[-3000:] if len(full_out) > 3000 else full_out)

dbutils.notebook.exit(str(exit_code))
'''

# Upload the runner notebook.
notebook_b64 = base64.b64encode(notebook_content.encode()).decode()
try:
    w.workspace.delete(path=f"{WORKSPACE_PATH}/run_tests", recursive=False)
except Exception:
    pass

w.workspace.import_(
    path=f"{WORKSPACE_PATH}/run_tests",
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
    content=notebook_b64,
)
print("Notebook uploaded.")

print("\nSubmitting test job...")

run_response = w.jobs.submit(
    run_name="insurance-rdd-tests",
    tasks=[
        jobs_service.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs_service.NotebookTask(
                notebook_path=f"{WORKSPACE_PATH}/run_tests",
            ),
        ),
    ],
)
run_id = run_response.run_id
print(f"Run submitted: {run_id}")

print("\nWaiting for tests...", flush=True)
task_run_id = None
for i in range(300):
    time.sleep(10)
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state.value if run_state.state.life_cycle_state else "UNKNOWN"
    result_state_val = run_state.state.result_state.value if run_state.state.result_state else "RUNNING"
    state_msg = (run_state.state.state_message or "")[:80]
    print(f"  [{(i+1)*10}s] {life_cycle} / {result_state_val}  {state_msg}", flush=True)

    if run_state.tasks and task_run_id is None:
        task_run_id = run_state.tasks[0].run_id

    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal state: {life_cycle} / {result_state_val}")

# Get notebook exit value.
notebook_exit_code = "unknown"
if task_run_id:
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output:
            notebook_exit_code = output.notebook_output.result
            print(f"Notebook exit value: {notebook_exit_code}")
        if output.error:
            print(f"Error: {output.error}")
        if output.error_trace:
            print(f"Error trace:\n{output.error_trace[-3000:]}")
    except Exception as e:
        print(f"Could not get output: {e}")

# Read test output from Workspace file.
print("\n=== Reading test output from Workspace ===")
try:
    exported = w.workspace.export(path=RESULTS_PATH)
    if exported.content:
        content = base64.b64decode(exported.content).decode("utf-8")
        print(content[-15000:] if len(content) > 15000 else content)
    else:
        print("No content in workspace file.")
except Exception as e:
    print(f"Could not read workspace file: {e}")

# Final verdict.
if notebook_exit_code == "0":
    print("\nALL TESTS PASSED")
    sys.exit(0)
elif result_state_val == "SUCCESS" and notebook_exit_code != "0":
    print(f"\nTESTS FAILED (exit code: {notebook_exit_code})")
    sys.exit(1)
else:
    print(f"\nJOB FAILED: {result_state_val}")
    sys.exit(1)
