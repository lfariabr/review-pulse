#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_gh(args: list[str]) -> str:
    process = subprocess.run(
        ["gh", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return process.stdout.strip()


def _discover_repo() -> str:
    return _run_gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _resolve_template_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if path.suffix.lower() == ".json":
        markdown_fallback = path.with_suffix(".md")
        if markdown_fallback.exists():
            print(
                f"[info] Template not found at {path}. Using markdown fallback {markdown_fallback}.",
                file=sys.stderr,
            )
            return markdown_fallback
    raise ValueError(f"Template file not found: {path}")


def _load_json_template(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Template file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON template: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Template root must be a JSON object.")
    return payload


def _clean_markdown_issue_body(section: str) -> str:
    cleaned = re.sub(r"^\s*---\s*$", "", section, flags=re.MULTILINE).strip()
    return cleaned


def _load_markdown_template(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^###\s+Issue\s+#(?P<number>\d+)\s*-\s*(?P<title>.+?)\s*$"
        r"(?P<section>[\s\S]*?)(?=^###\s+Issue\s+#\d+\s*-|^##\s+|\Z)",
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(content))
    if not matches:
        raise ValueError(
            "Markdown template must contain headings like: '### Issue #50 - Your title'."
        )

    issues: list[dict[str, Any]] = []
    for match in matches:
        issue_number = match.group("number")
        title = match.group("title").strip()
        section = _clean_markdown_issue_body(match.group("section"))
        body_content = section if section else "_No body content provided in markdown section._"
        body = f"Source: `{path.as_posix()}` (Issue #{issue_number})\n\n{body_content}"
        issues.append(
            {
                "title": title,
                "body": body,
                "labels": [],
                "assignees": [],
                "milestone": None,
            }
        )

    return {"issues": issues, "defaults": {"labels": []}}


def _load_template(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".md":
        return _load_markdown_template(path)
    return _load_json_template(path)


def _validate_issues(raw_issues: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_issues, list) or not raw_issues:
        raise ValueError("Template must contain a non-empty 'issues' list.")

    validated: list[dict[str, Any]] = []
    for idx, issue in enumerate(raw_issues, start=1):
        if not isinstance(issue, dict):
            raise ValueError(f"Issue #{idx} must be a JSON object.")

        title = issue.get("title", "")
        body = issue.get("body", "")
        labels = issue.get("labels", [])
        assignees = issue.get("assignees", [])
        milestone = issue.get("milestone")

        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Issue #{idx} must include a non-empty 'title'.")
        if not isinstance(body, str):
            raise ValueError(f"Issue #{idx} field 'body' must be a string.")
        if not isinstance(labels, list) or any(not isinstance(v, str) for v in labels):
            raise ValueError(f"Issue #{idx} field 'labels' must be a list of strings.")
        if not isinstance(assignees, list) or any(not isinstance(v, str) for v in assignees):
            raise ValueError(f"Issue #{idx} field 'assignees' must be a list of strings.")
        if milestone is not None and not isinstance(milestone, str):
            raise ValueError(f"Issue #{idx} field 'milestone' must be a string when provided.")

        validated.append(
            {
                "title": title.strip(),
                "body": body,
                "labels": labels,
                "assignees": assignees,
                "milestone": milestone,
            }
        )

    return validated


def _existing_titles(repo: str) -> dict[str, int]:
    data = _run_gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "title,number",
        ]
    )
    issues = json.loads(data)
    return {issue["title"]: issue["number"] for issue in issues}


def _build_create_cmd(
    repo: str,
    title: str,
    body: str,
    labels: list[str],
    assignees: list[str],
    milestone: str | None,
) -> list[str]:
    cmd = [
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body",
        body,
    ]
    for label in labels:
        cmd.extend(["--label", label])
    for assignee in assignees:
        cmd.extend(["--assignee", assignee])
    if milestone:
        cmd.extend(["--milestone", milestone])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-create GitHub issues from JSON templates or markdown issue breakdowns."
    )
    parser.add_argument(
        "--template",
        default="docs/templates/issue_creator.template.json",
        help="Path to template file (.json or markdown issue breakdown .md).",
    )
    parser.add_argument("--repo", help="Target repo in owner/name format.")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Actually create issues on GitHub. Defaults to dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created (default behavior).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip issues when the same title already exists (default: enabled).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip duplicates by title.",
    )
    args = parser.parse_args()

    try:
        template_path = _resolve_template_path(args.template)
        payload = _load_template(template_path)
        issues = _validate_issues(payload.get("issues"))
        repo = args.repo or payload.get("repo") or _discover_repo()
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError("Unable to determine repository. Use --repo or set 'repo' in template.")
        repo = repo.strip()
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        print("[error] 'defaults' must be an object when provided.", file=sys.stderr)
        return 1

    global_labels = defaults.get("labels", [])
    if not isinstance(global_labels, list) or any(not isinstance(v, str) for v in global_labels):
        print("[error] 'defaults.labels' must be a list of strings when provided.", file=sys.stderr)
        return 1

    existing_titles: dict[str, int] = {}
    if args.skip_existing:
        try:
            existing_titles = _existing_titles(repo)
        except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            print(f"[error] Failed loading existing issues: {exc}", file=sys.stderr)
            return 1

    created: list[str] = []
    skipped: list[str] = []
    planned: list[str] = []

    for issue in issues:
        title = issue["title"]
        body = issue["body"]
        labels = _dedupe([*global_labels, *issue["labels"]])
        assignees = _dedupe(issue["assignees"])
        milestone = issue["milestone"]

        if args.skip_existing and title in existing_titles:
            skipped.append(f"- {title} (already exists: #{existing_titles[title]})")
            continue

        cmd = _build_create_cmd(repo, title, body, labels, assignees, milestone)

        if not args.create:
            details = []
            if labels:
                details.append(f"labels={labels}")
            if assignees:
                details.append(f"assignees={assignees}")
            if milestone:
                details.append(f"milestone={milestone}")
            suffix = f" [{', '.join(details)}]" if details else ""
            planned.append(f"- {title}{suffix}")
            continue

        try:
            url = _run_gh(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"[error] Failed creating '{title}': {exc.stderr.strip()}", file=sys.stderr)
            return 1
        created.append(f"- {title} -> {url}")

    mode = "CREATE" if args.create else "DRY-RUN"
    print(f"[{mode}] repo={repo}")
    if planned:
        print("Planned:")
        print("\n".join(planned))
    if created:
        print("Created:")
        print("\n".join(created))
    if skipped:
        print("Skipped:")
        print("\n".join(skipped))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
