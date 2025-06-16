import subprocess
import shlex

def git_commit(message: str) -> dict:
    """
    Creates a Git commit with the given message.
    Adds all changed files before committing.
    """
    try:
        # Stage all changes
        subprocess.run(["git", "add", "."], check=True)
        # Commit with the provided message
        result = subprocess.run(
            ["git", "commit", "-m", message],
            check=True,
            capture_output=True,
            text=True
        )
        return {"status": "success", "output": result.stdout}
    except FileNotFoundError:
        return {"status": "error", "reason": "Git command not found. Is Git installed and in your PATH?"}
    except subprocess.CalledProcessError as e:
        # This error is common if there's nothing to commit
        return {"status": "error", "reason": f"Git commit failed: {e.stderr}"}

def git_status() -> dict:
    """
    Checks the current status of the Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "status"],
            check=True,
            capture_output=True,
            text=True
        )
        return {"status": "success", "output": result.stdout}
    except FileNotFoundError:
        return {"status": "error", "reason": "Git command not found. Is Git installed and in your PATH?"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "reason": f"Git status failed: {e.stderr}"}
