import subprocess
import tempfile
import json
from pathlib import Path

class DockerSandbox:
    """Execute agent commands inside a restricted Docker container."""

    def __init__(
        self,
        image: str = "agent-sandbox:latest",
        workspace: str | None = None,
        timeout: int = 30,
        memory_limit: str = "512m",
        network: bool = False,
    ):
        self.image = image
        self.workspace = workspace or tempfile.mkdtemp(prefix="agent-")
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.network = network

    def execute(self, command: str) -> dict:
        """
        Execute a command in an isolated Docker sandbox.

        Args:
            command (str): Shell command to execute.

        Returns:
            dict: Execution results containing stdout, stderr, and exit code.
        """
        docker_cmd = [
            "docker", "run",
            "--rm",                              # Auto-cleanup
            "--user", "1000:1000",               # Non-root
            "--memory", self.memory_limit,       # OOM protection
            "--cpus", "1.0",                     # CPU limit
            "--pids-limit", "100",               # Fork bomb protection
            "--read-only",                       # Read-only root filesystem
            "--tmpfs", "/tmp:size=100m",         # Writable temp space
            "--tmpfs", "/workspace:size=200m",   # Writable workspace
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",                 # Drop all Linux capabilities
        ]

        # Mount workspace files as read-only input
        if Path(self.workspace).exists():
            docker_cmd.extend([
                "-v", f"{self.workspace}:/input:ro"
            ])

        # Network isolation (default: no network)
        if not self.network:
            docker_cmd.extend(["--network", "none"])

        docker_cmd.extend([self.image, "bash", "-c", command])

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "stdout": result.stdout[-10_000:],  # Truncate large output
                "stderr": result.stderr[-5_000:],
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return json.dumps({
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout}s",
                "exit_code": -1,
            })