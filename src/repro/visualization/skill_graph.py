"""Render the upstream Markdown skill grammar without inventing control flow."""

from __future__ import annotations

import html
import re

_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3 = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def _label(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return (
        html.escape(normalized, quote=True)
        .replace("|", "&#124;")
        .replace("`", "&#96;")
        .replace("\n", "<br/>")
    )


def _check_embedded_mermaid(diagram: str) -> None:
    """Keep plain flowcharts verbatim but reject active renderer instructions.

    Exported artifacts are text, not HTML. If rendered later, the renderer must
    retain Mermaid's strict security mode; this check is not an HTML sanitizer.
    """
    if not re.match(r"\s*(?:flowchart|graph)\s+(?:TD|TB|BT|LR|RL)\b", diagram):
        raise ValueError("Embedded Mermaid must be a plain flowchart or graph")
    decoded = html.unescape(diagram)
    unsafe = (
        r"%%\s*\{", r"\bsecurityLevel\b", r"(?:^|[;\n])\s*click\s+",
        r"\b(?:javascript|vbscript)\s*:", r"\bdata\s*:\s*text/html", r"\burl\s*\(",
        r"<\s*(?:script|iframe|object|embed|img|svg|link|style|a)\b", r"\bon\w+\s*=",
    )
    if any(re.search(pattern, decoded, re.IGNORECASE | re.MULTILINE) for pattern in unsafe):
        raise ValueError("Embedded Mermaid contains active content or renderer configuration")


def _sections(content: str, pattern: re.Pattern[str]) -> dict[str, str]:
    matches = list(pattern.finditer(content))
    parts = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name in parts:
            raise ValueError(f"Duplicate skill heading: {name}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        lines = content[match.end():end].splitlines()
        while lines and lines[0].strip() in {"", "---"}:
            lines.pop(0)
        while lines and lines[-1].strip() in {"", "---"}:
            lines.pop()
        parts[name] = "\n".join(lines).strip()
    return parts


def skill_to_mermaid(content: str) -> str:
    """Export explicit routing and numbered node order to one Mermaid flowchart.

    Existing Mermaid is kept verbatim. In the upstream Markdown grammar, task
    ``Use when`` conditions become routing edges, and full node instructions
    preserve any natural-language conditions. We do not infer branch edges from
    prose: workflows only declare node order. Route-specific node instances
    avoid introducing spurious paths when workflows share reusable nodes.
    Full node guidance appears once in a library, linked by numbered labels;
    repeating long guidance in every route would make practical diagrams huge.
    """
    existing = _FENCE.findall(content)
    if existing:
        if len(existing) != 1:
            raise ValueError("Skill contains multiple Mermaid blocks; export each block separately")
        _check_embedded_mermaid(existing[0])
        return existing[0].strip() + "\n"
    sections = _sections(content, _H2)
    if not {"Global Guidance", "Node Lists", "Task Graphs"}.issubset(sections):
        raise ValueError("Skill must contain Global Guidance, Node Lists, and Task Graphs sections")
    nodes = _sections(sections["Node Lists"], _H3)
    tasks = _sections(sections["Task Graphs"], _H3)
    node_numbers = {name: index + 1 for index, name in enumerate(nodes)}
    lines = ["flowchart TD", '    start(["Task input"])']
    guidance = sections["Global Guidance"].strip()
    lines.append(f'    guidance["Global Guidance<br/>{_label(guidance)}"]')
    lines.append("    start --> guidance")
    if not tasks:
        if nodes:
            raise ValueError("Skill defines nodes but no task workflows")
        lines.extend(['    empty["No nodes or task workflows declared"]', "    guidance --> empty"])
        return "\n".join(lines) + "\n"
    lines.extend(['    route{"Task type routing"}', "    guidance --> route"])
    lines.append('    subgraph node_guidance["Reusable node instructions; N labels match workflow uses"]')
    for name, body in nodes.items():
        number = node_numbers[name]
        text = f"N{number}: {name}" + ("\n" + body if body else "")
        lines.append(f'        guide_n{number}["{_label(text)}"]')
    lines.append("    end")
    referenced: set[str] = set()
    for task_index, (title, body) in enumerate(tasks.items()):
        condition = re.search(r"^\*\*Use when:\*\*\s*(.+)$", body, re.MULTILINE)
        workflow = re.search(r"^\*\*Workflow:\*\*\s*$", body, re.MULTILINE)
        if condition is None or workflow is None or condition.start() >= workflow.start():
            raise ValueError(f"Task '{title}' needs Use when followed by Workflow")
        items = []
        for line in body[workflow.end():].splitlines():
            if not line.strip() or line.strip() == "---":
                continue
            match = re.fullmatch(r"(\d+)\.\s+(.+)", line.strip())
            if match is None or int(match.group(1)) != len(items) + 1:
                raise ValueError(f"Task '{title}' needs consecutive numbered workflow nodes")
            name = match.group(2).strip()
            if name not in nodes:
                raise ValueError(f"Task '{title}' references undefined node '{name}'")
            items.append(name)
            referenced.add(name)
        if not items:
            raise ValueError(f"Task '{title}' has an empty workflow")
        task_id = f"task_{task_index}"
        lines.append(f'    {task_id}["{_label(title)}"]')
        lines.append(f'    route -->|"{_label(condition.group(1).strip())}"| {task_id}')
        previous = task_id
        for step, name in enumerate(items):
            node_id = f"t{task_index}_n{step}"
            text = f"N{node_numbers[name]}: {name}"
            lines.append(f'    {node_id}["{_label(text)}"]')
            lines.append(f"    {previous} --> {node_id}")
            previous = node_id
        lines.append(f'    t{task_index}_end(["Complete task"])')
        lines.append(f"    {previous} --> t{task_index}_end")
    unused = nodes.keys() - referenced
    if unused:
        raise ValueError(f"Skill contains unused nodes: {sorted(unused)}")
    return "\n".join(lines) + "\n"
