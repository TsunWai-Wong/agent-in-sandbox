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
        # The workspace is bind-mounted as the container's working directory, so
        # it has to exist before `docker run`. Left missing, Docker creates the
        # source path itself owned by root -- and the container runs as uid
        # 1000, which then cannot write to its own working directory.
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.network = network

    def execute(self, command: str) -> str:
        """
        Execute a command in an isolated Docker sandbox.

        Args:
            command (str): Shell command to execute.

        Returns:
            str: JSON object with keys stdout, stderr and exit_code.
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
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",                 # Drop all Linux capabilities
        ]

        # The workspace is a read-write bind mount used as the working
        # directory, not a tmpfs. A tmpfs dies with its container and every
        # execute() starts a fresh one, so a file written by one call was gone
        # by the next and nothing multi-step could work. Being the only
        # writable path that outlives a call is also what makes this the
        # exchange point between the host and the agent.
        docker_cmd.extend([
            "-v", f"{self.workspace}:/workspace",
            "-w", "/workspace",
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
            # JSON rather than the dict itself: ToolRegistry.dispatch stringifies
            # whatever a tool returns with str(), and str(dict) is Python repr --
            # single-quoted and not parseable as JSON by the model reading it.
            return json.dumps({
                "stdout": result.stdout[-10_000:],  # Truncate large output
                "stderr": result.stderr[-5_000:],
                "exit_code": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout}s",
                "exit_code": -1,
            })