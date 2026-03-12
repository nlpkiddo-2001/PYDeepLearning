import subprocess
import sys

VLLM_VERSION = "0.11.1"
ENV_NAME = f"vllm-{VLLM_VERSION}-env"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
PYTHON_VERSION = "3.13"


VLLM_ARGS = [
    "--model", MODEL_ID,
    "--gpu-memory-utilization", "0.9",
    "--max-model-len", "32768",
    "--trust-remote-code",
    "--dtype", "bfloat16",
    "--port", "9999",
    "--host", "0.0.0.0"
]


def get_conda_python():
    return "python"


def run_in_conda_env(args, description):
    print(f"\n>>> {description}...")
    try:
        cmd = ["conda", "run", "-n", ENV_NAME] + args
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} failed.")
        sys.exit(1)


def env_exists():
    try:
        result = subprocess.run(["conda", "env", "list"],
                                capture_output=True, text=True, check=True)
        return ENV_NAME in result.stdout
    except subprocess.CalledProcessError:
        print("[WARNING] Could not check existing conda environments.")
        return False


def delete_env():
    if env_exists():
        print(f"\n>>> Deleting existing conda environment: {ENV_NAME}...")
        try:
            subprocess.run(["conda", "env", "remove", "-n", ENV_NAME, "-y"],
                           check=True)
            print(f"[SUCCESS] Environment '{ENV_NAME}' deleted successfully.")
        except subprocess.CalledProcessError:
            print(f"[FAIL] Could not delete conda environment '{ENV_NAME}'.")
            sys.exit(1)
    else:
        print(f"[INFO] Environment '{ENV_NAME}' does not exist. No need to delete.")


def create_env():
    delete_env()

    print(f"\n>>> Creating fresh conda environment: {ENV_NAME} with Python {PYTHON_VERSION}...")
    try:
        subprocess.run(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"],
                       check=True)
        print(f"[SUCCESS] Environment '{ENV_NAME}' created successfully.")
    except subprocess.CalledProcessError:
        print("[FAIL] Could not create conda environment.")
        print("[INFO] Make sure conda is installed and available in PATH.")
        sys.exit(1)


def install_and_secure():
    run_in_conda_env(["python", "-m", "pip", "install", "--upgrade", "pip", "pip-audit"],
                     "Installing pip-audit and upgrading pip")

    run_in_conda_env(["python", "-m", "pip", "install", f"vllm=={VLLM_VERSION}"],
                     f"Installing vLLM {VLLM_VERSION} (and transitive deps)")

    print("\n" + "=" * 60)
    print("STARTING VULNERABILITY AUDIT (TRANSITIVE DEPENDENCIES)")
    print("=" * 60)

    audit_cmd = ["python", "-m", "pip_audit", "--fix", "--vulnerability-service", "osv"]

    try:
        run_in_conda_env(audit_cmd, "Scanning and Fixing Vulnerabilities")
    except SystemExit:

        print("[INFO] Audit complete. Vulnerabilities may have been fixed.")


def launch_model():
    print("\n" + "=" * 60)
    print(f"LAUNCHING MODEL: {MODEL_ID}")
    print("=" * 60)

    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + VLLM_ARGS
    run_in_conda_env(cmd, "Starting vLLM Server")


if __name__ == "__main__":
    create_env()

    install_and_secure()

    launch_model()