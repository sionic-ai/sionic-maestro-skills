"""
Workspace management for safe file operations and patch application.

Implements safe file modification following the principle:
"Tool execution (file write/command execution) is restricted to a single Executor"

Key safety features:
- Path validation (no escape from workspace root)
- Allowlist-based file patterns
- Backup before modification
- Atomic writes where possible
- Unified diff parsing and application
"""

import difflib
import hashlib
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger("maestro.workspace")


@dataclass
class FileChange:
    """Represents a single file change."""
    path: str
    change_type: str  # 'create', 'modify', 'delete', 'rename'
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: Optional[str] = None
    old_path: Optional[str] = None  # For renames


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    files_changed: List[str]
    files_created: List[str]
    files_failed: List[str]
    backup_dir: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_changed": self.files_changed,
            "files_created": self.files_created,
            "files_failed": self.files_failed,
            "backup_dir": self.backup_dir,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class WorkspaceConfig:
    """Configuration for workspace operations."""
    # Allowed file patterns (glob-style)
    allowed_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
        "*.go", "*.rs", "*.java", "*.kt",
        "*.c", "*.cpp", "*.h", "*.hpp",
        "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml",
        "*.html", "*.css", "*.scss",
        "Makefile", "Dockerfile", "*.dockerfile",
        "*.sh", "*.bash",
    ])

    # Blocked patterns (never modify these)
    blocked_patterns: List[str] = field(default_factory=lambda: [
        "*.env", ".env*",
        "*.pem", "*.key", "*.crt",
        "*credentials*", "*secret*",
        ".git/*", ".git/**",
        "node_modules/*", "node_modules/**",
        "__pycache__/*", "*.pyc",
        ".venv/*", "venv/*",
    ])

    # Max file size to modify (bytes)
    max_file_size: int = 1_000_000  # 1MB

    # Create backups before modification
    create_backups: bool = True

    # Backup directory (relative to workspace root)
    backup_dir: str = ".maestro-backups"


class WorkspaceManager:
    """
    Manages file operations within a workspace with safety constraints.

    Security principles:
    - All paths are validated against workspace root
    - File patterns are checked against allowlist/blocklist
    - Backups are created before modifications
    - Atomic writes where possible
    """

    def __init__(
        self,
        workspace_root: str,
        config: Optional[WorkspaceConfig] = None,
    ):
        self.root = Path(workspace_root).resolve()
        self.config = config or WorkspaceConfig()
        self._backup_session_id: Optional[str] = None

    def _validate_path(self, path: str) -> Tuple[bool, str, Path]:
        """
        Validate that a path is safe to access.

        Returns:
            (is_valid, error_message, resolved_path)
        """
        try:
            # Resolve to absolute path
            if os.path.isabs(path):
                resolved = Path(path).resolve()
            else:
                resolved = (self.root / path).resolve()

            # Check if within workspace
            try:
                resolved.relative_to(self.root)
            except ValueError:
                return False, f"Path escapes workspace root: {path}", resolved

            # Check against blocked patterns
            rel_path = str(resolved.relative_to(self.root))
            for pattern in self.config.blocked_patterns:
                if self._matches_pattern(rel_path, pattern):
                    return False, f"Path matches blocked pattern '{pattern}': {path}", resolved

            return True, "", resolved

        except Exception as e:
            return False, f"Invalid path: {e}", Path(path)

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a glob-style pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)

    def _is_allowed_file(self, path: str) -> bool:
        """Check if file matches allowed patterns."""
        basename = os.path.basename(path)
        for pattern in self.config.allowed_patterns:
            if self._matches_pattern(path, pattern) or self._matches_pattern(basename, pattern):
                return True
        return False

    def _create_backup(self, file_path: Path) -> Optional[str]:
        """Create a backup of a file before modification."""
        if not self.config.create_backups or not file_path.exists():
            return None

        # Create session backup directory if needed
        if self._backup_session_id is None:
            self._backup_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_dir = self.root / self.config.backup_dir / self._backup_session_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Preserve directory structure in backup
        rel_path = file_path.relative_to(self.root)
        backup_path = backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def read_file(self, path: str) -> Tuple[bool, str, Optional[str]]:
        """
        Read a file from the workspace.

        Returns:
            (success, error_or_content, content_if_success)
        """
        is_valid, error, resolved = self._validate_path(path)
        if not is_valid:
            return False, error, None

        if not resolved.exists():
            return False, f"File not found: {path}", None

        if not resolved.is_file():
            return False, f"Not a file: {path}", None

        # Check file size
        if resolved.stat().st_size > self.config.max_file_size:
            return False, f"File too large: {resolved.stat().st_size} bytes", None

        try:
            content = resolved.read_text(encoding="utf-8")
            return True, content, content
        except UnicodeDecodeError:
            return False, "File is not valid UTF-8 text", None
        except Exception as e:
            return False, f"Error reading file: {e}", None

    def write_file(
        self,
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> Tuple[bool, str]:
        """
        Write content to a file.

        Returns:
            (success, error_or_backup_path)
        """
        is_valid, error, resolved = self._validate_path(path)
        if not is_valid:
            return False, error

        if not self._is_allowed_file(str(resolved)):
            return False, f"File type not allowed: {path}"

        # Create backup if file exists
        backup_path = None
        if resolved.exists():
            backup_path = self._create_backup(resolved)

        try:
            # Create parent directories if needed
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write using temp file
            fd, temp_path = tempfile.mkstemp(
                dir=resolved.parent,
                prefix=".maestro_tmp_",
                suffix=resolved.suffix,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(temp_path, resolved)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

            return True, backup_path or "success"

        except Exception as e:
            return False, f"Error writing file: {e}"

    def delete_file(self, path: str) -> Tuple[bool, str]:
        """Delete a file (with backup)."""
        is_valid, error, resolved = self._validate_path(path)
        if not is_valid:
            return False, error

        if not resolved.exists():
            return True, "File already deleted"

        # Create backup
        backup_path = self._create_backup(resolved)

        try:
            resolved.unlink()
            return True, backup_path or "deleted"
        except Exception as e:
            return False, f"Error deleting file: {e}"

    def apply_patch(
        self,
        patch_content: str,
        dry_run: bool = False,
    ) -> PatchResult:
        """
        Apply a unified diff patch to the workspace.

        Supports:
        - Standard unified diff format
        - Git diff format
        - Multiple files in one patch

        Args:
            patch_content: The unified diff content
            dry_run: If True, validate but don't apply

        Returns:
            PatchResult with details of what was changed
        """
        changes = self._parse_unified_diff(patch_content)

        if not changes:
            return PatchResult(
                success=False,
                files_changed=[],
                files_created=[],
                files_failed=[],
                error="No valid changes found in patch",
            )

        files_changed = []
        files_created = []
        files_failed = []
        details = {}

        # Start backup session
        if self.config.create_backups and not dry_run:
            self._backup_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = str(self.root / self.config.backup_dir / self._backup_session_id)
        else:
            backup_dir = None

        for change in changes:
            path = change.path
            is_valid, error, resolved = self._validate_path(path)

            if not is_valid:
                files_failed.append(path)
                details[path] = {"error": error, "status": "blocked"}
                continue

            if not self._is_allowed_file(path):
                files_failed.append(path)
                details[path] = {"error": "File type not allowed", "status": "blocked"}
                continue

            if dry_run:
                # Just validate
                if change.change_type == "create":
                    files_created.append(path)
                else:
                    files_changed.append(path)
                details[path] = {"status": "would_apply"}
                continue

            # Apply the change
            try:
                if change.change_type == "create":
                    if change.new_content is not None:
                        success, msg = self.write_file(path, change.new_content)
                        if success:
                            files_created.append(path)
                            details[path] = {"status": "created"}
                        else:
                            files_failed.append(path)
                            details[path] = {"error": msg, "status": "failed"}

                elif change.change_type == "delete":
                    success, msg = self.delete_file(path)
                    if success:
                        files_changed.append(path)
                        details[path] = {"status": "deleted", "backup": msg}
                    else:
                        files_failed.append(path)
                        details[path] = {"error": msg, "status": "failed"}

                elif change.change_type == "modify":
                    # Read current content
                    success, content_or_error, current_content = self.read_file(path)
                    if not success:
                        # File doesn't exist - treat as create
                        if change.new_content is not None:
                            success, msg = self.write_file(path, change.new_content)
                            if success:
                                files_created.append(path)
                                details[path] = {"status": "created"}
                            else:
                                files_failed.append(path)
                                details[path] = {"error": msg, "status": "failed"}
                        continue

                    # Apply diff to get new content
                    if change.new_content is not None:
                        new_content = change.new_content
                    elif change.diff:
                        new_content = self._apply_diff_to_content(current_content, change.diff)
                    else:
                        files_failed.append(path)
                        details[path] = {"error": "No new content or diff", "status": "failed"}
                        continue

                    if new_content is None:
                        files_failed.append(path)
                        details[path] = {"error": "Diff application failed", "status": "failed"}
                        continue

                    success, msg = self.write_file(path, new_content)
                    if success:
                        files_changed.append(path)
                        details[path] = {"status": "modified", "backup": msg}
                    else:
                        files_failed.append(path)
                        details[path] = {"error": msg, "status": "failed"}

            except Exception as e:
                files_failed.append(path)
                details[path] = {"error": str(e), "status": "exception"}

        success = len(files_failed) == 0 and (len(files_changed) > 0 or len(files_created) > 0)

        return PatchResult(
            success=success,
            files_changed=files_changed,
            files_created=files_created,
            files_failed=files_failed,
            backup_dir=backup_dir,
            error=None if success else f"{len(files_failed)} files failed",
            details=details,
        )

    def _parse_unified_diff(self, diff_content: str) -> List[FileChange]:
        """
        Parse a unified diff into FileChange objects.

        Handles:
        - Standard unified diff (--- a/file, +++ b/file)
        - Git diff format
        - Simple file creation (no --- line)
        """
        changes = []
        lines = diff_content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for diff header
            if line.startswith("diff --git"):
                # Git format: diff --git a/file b/file
                match = re.match(r'diff --git a/(.+) b/(.+)', line)
                if match:
                    old_path = match.group(1)
                    new_path = match.group(2)

                    # Collect the diff content for this file
                    i += 1
                    diff_lines = [line]
                    while i < len(lines) and not lines[i].startswith("diff --git"):
                        diff_lines.append(lines[i])
                        i += 1

                    diff_text = "\n".join(diff_lines)

                    # Determine change type
                    if "new file mode" in diff_text:
                        change_type = "create"
                    elif "deleted file mode" in diff_text:
                        change_type = "delete"
                    else:
                        change_type = "modify"

                    # Extract new content from + lines
                    new_content = self._extract_new_content(diff_text)

                    changes.append(FileChange(
                        path=new_path,
                        change_type=change_type,
                        diff=diff_text,
                        new_content=new_content,
                        old_path=old_path if old_path != new_path else None,
                    ))
                    continue

            # Standard unified diff format
            elif line.startswith("--- "):
                old_file_match = re.match(r'--- (?:a/)?(.+)', line)
                if old_file_match and i + 1 < len(lines):
                    new_file_line = lines[i + 1]
                    new_file_match = re.match(r'\+\+\+ (?:b/)?(.+)', new_file_line)

                    if new_file_match:
                        old_path = old_file_match.group(1).strip()
                        new_path = new_file_match.group(1).strip()

                        # Handle /dev/null for creates/deletes
                        if old_path == "/dev/null":
                            change_type = "create"
                            path = new_path
                        elif new_path == "/dev/null":
                            change_type = "delete"
                            path = old_path
                        else:
                            change_type = "modify"
                            path = new_path

                        # Collect diff hunks
                        i += 2
                        diff_lines = [line, new_file_line]
                        while i < len(lines):
                            if lines[i].startswith("--- ") or lines[i].startswith("diff --git"):
                                break
                            diff_lines.append(lines[i])
                            i += 1

                        diff_text = "\n".join(diff_lines)
                        new_content = self._extract_new_content(diff_text)

                        changes.append(FileChange(
                            path=path,
                            change_type=change_type,
                            diff=diff_text,
                            new_content=new_content,
                        ))
                        continue

            i += 1

        return changes

    def _extract_new_content(self, diff_text: str) -> Optional[str]:
        """Extract the complete new file content from a diff."""
        lines = diff_text.split("\n")
        new_lines = []
        in_hunk = False

        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue

            if in_hunk:
                if line.startswith("+") and not line.startswith("+++"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    new_lines.append(line[1:])
                elif line.startswith("-"):
                    continue  # Skip removed lines
                elif line.startswith("\\"):
                    continue  # Skip "\ No newline at end of file"

        if new_lines:
            return "\n".join(new_lines)
        return None

    def _apply_diff_to_content(self, original: str, diff_text: str) -> Optional[str]:
        """Apply a unified diff to original content."""
        try:
            # Parse hunks from diff
            original_lines = original.split("\n")
            result_lines = original_lines.copy()

            # Find hunks: @@ -start,count +start,count @@
            hunk_pattern = re.compile(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
            lines = diff_text.split("\n")

            offset = 0  # Track line number offset as we modify

            i = 0
            while i < len(lines):
                match = hunk_pattern.match(lines[i])
                if match:
                    old_start = int(match.group(1)) - 1  # 0-indexed
                    new_start = int(match.group(3)) - 1

                    # Collect hunk lines
                    i += 1
                    removals = []
                    additions = []
                    context_before = []

                    while i < len(lines):
                        line = lines[i]
                        if line.startswith("@@") or line.startswith("diff ") or line.startswith("--- "):
                            break
                        if line.startswith("-") and not line.startswith("---"):
                            removals.append(line[1:])
                        elif line.startswith("+") and not line.startswith("+++"):
                            additions.append(line[1:])
                        elif line.startswith(" "):
                            if not removals and not additions:
                                context_before.append(line[1:])
                        i += 1

                    # Apply hunk
                    actual_start = old_start + offset
                    # Remove old lines
                    for _ in range(len(removals)):
                        if actual_start < len(result_lines):
                            result_lines.pop(actual_start)
                    # Insert new lines
                    for j, add_line in enumerate(additions):
                        result_lines.insert(actual_start + j, add_line)

                    offset += len(additions) - len(removals)
                else:
                    i += 1

            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Failed to apply diff: {e}")
            return None

    def get_file_hash(self, path: str) -> Optional[str]:
        """Get SHA256 hash of a file."""
        is_valid, error, resolved = self._validate_path(path)
        if not is_valid or not resolved.exists():
            return None

        try:
            content = resolved.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return None

    def list_modified_files(self, since_backup_session: Optional[str] = None) -> List[str]:
        """List files that have been modified (have backups)."""
        backup_base = self.root / self.config.backup_dir

        if since_backup_session:
            backup_dir = backup_base / since_backup_session
        elif self._backup_session_id:
            backup_dir = backup_base / self._backup_session_id
        else:
            return []

        if not backup_dir.exists():
            return []

        modified = []
        for backup_file in backup_dir.rglob("*"):
            if backup_file.is_file():
                rel_path = str(backup_file.relative_to(backup_dir))
                modified.append(rel_path)

        return modified

    def restore_from_backup(
        self,
        backup_session: str,
        files: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Restore files from a backup session.

        Returns:
            (success, restored_files, failed_files)
        """
        backup_dir = self.root / self.config.backup_dir / backup_session

        if not backup_dir.exists():
            return False, [], [f"Backup session not found: {backup_session}"]

        restored = []
        failed = []

        for backup_file in backup_dir.rglob("*"):
            if not backup_file.is_file():
                continue

            rel_path = str(backup_file.relative_to(backup_dir))

            if files and rel_path not in files:
                continue

            target = self.root / rel_path

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, target)
                restored.append(rel_path)
            except Exception as e:
                failed.append(f"{rel_path}: {e}")

        return len(failed) == 0, restored, failed
