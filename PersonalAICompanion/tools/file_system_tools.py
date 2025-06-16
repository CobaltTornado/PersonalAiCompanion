import os
import logging

# Set up logging for file system tools
FS_LOGGER = logging.getLogger("FileSystemTools")

def create_file(file_path: str, content: str = "") -> dict:
    """
    Creates a new file with the specified content directly on the file system.
    This is an autonomous operation.
    """
    FS_LOGGER.info(f"TOOL: Creating new file at {file_path}")
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {
            "status": "success",
            "message": f"Successfully created file: {file_path}"
        }
    except Exception as e:
        error_message = f"Failed to create file: {str(e)}"
        FS_LOGGER.error(error_message)
        return {"status": "error", "reason": error_message}

def read_file(file_path: str) -> dict:
    """
    Reads the content of a file from the local file system.
    This is one of an agent's primary ways of understanding the project state.
    """
    FS_LOGGER.info(f"TOOL: Reading file at {file_path}")
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "reason": f"File not found at {file_path}"}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"status": "success", "content": content}
    except Exception as e:
        error_message = f"Failed to read file: {str(e)}"
        FS_LOGGER.error(error_message)
        return {"status": "error", "reason": error_message}

def update_file(file_path: str, content: str) -> dict:
    """
    Updates a file with the new content directly on the file system.
    This is an autonomous operation.
    """
    FS_LOGGER.info(f"TOOL: Updating file at {file_path}")
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "reason": f"File not found at {file_path}"}
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {
            "status": "success",
            "message": f"Successfully updated file: {file_path}"
        }
    except Exception as e:
        error_message = f"Failed to update file: {str(e)}"
        FS_LOGGER.error(error_message)
        return {"status": "error", "reason": error_message}

def get_fs_state_str() -> str:
    """
    Provides a string representation of the file system state.
    (This is a placeholder for a more complex implementation if needed)
    """
    # In a real-world scenario, this might walk the directory tree.
    # For now, we'll keep it simple as the agent primarily uses read_file.
    return "File system state is actively managed by file read/write tools."

