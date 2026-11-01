"""Workflow diagrams must preserve declared routes and avoid invented branches."""

import pytest

from repro.visualization import lineage_to_mermaid, skill_to_mermaid

SKILL = """# Demo
## Global Guidance
Use the supplied evidence.
## Node Lists
### Read
If the evidence is missing, state uncertainty.
### Answer
Return <answer>text</answer>.
## Task Graphs
### Direct extraction
**Use when:** The answer occurs in one sentence.
**Workflow:**
1. Read
2. Answer
### Multi-hop
**Use when:** More than one supporting sentence is needed.
**Workflow:**
1. Read
2. Answer
"""


def test_workflows_show_routes_node_guidance_and_route_specific_edges():
    diagram = skill_to_mermaid(SKILL)
    assert "Task type routing" in diagram
    assert "The answer occurs in one sentence." in diagram
    assert "If the evidence is missing, state uncertainty." in diagram
    assert "t0_n0 --> t0_n1" in diagram
    assert "t1_n0 --> t1_n1" in diagram
    assert "t0_n0 --> t1_n1" not in diagram
    assert "&lt;answer&gt;text&lt;/answer&gt;" in diagram


def test_existing_mermaid_conditions_are_preserved_verbatim():
    original = 'flowchart TD\nA{"Enough evidence?"} -->|yes| B[Answer]\nA -->|no| C[Search]'
    assert skill_to_mermaid(f"```mermaid\n{original}\n```\n") == original + "\n"


def test_empty_initial_graph_is_explicit_and_undefined_nodes_fail():
    empty = "## Global Guidance\nFollow instructions.\n## Node Lists\n\n## Task Graphs\n"
    assert "No nodes or task workflows declared" in skill_to_mermaid(empty)
    with pytest.raises(ValueError, match="undefined"):
        skill_to_mermaid(SKILL.replace("2. Answer", "2. Unknown"))


def test_lineage_uses_parent_ids_without_claiming_execution_edges():
    diagram = lineage_to_mermaid([
        {"skill_id": "initial", "generation": 0, "parent_ids": []},
        {"skill_id": "child", "generation": 1, "parent_ids": ["initial"], "val_hard": 0.5},
    ])
    assert "s0 --> s1" in diagram
    assert "generation: 1" in diagram
    assert "val_hard: 0.5" in diagram
    assert "Task type routing" not in diagram


def test_prose_keeps_bullets_and_cannot_inject_mermaid_structure():
    malicious = 'Read"]\r\nInjected --> X["x'
    skill = SKILL.replace("### Read\n", f"### {malicious.replace(chr(13), '').replace(chr(10), ' ')}\n")
    skill = skill.replace("1. Read", f"1. {malicious.replace(chr(13), '').replace(chr(10), ' ')}")
    diagram = skill_to_mermaid(skill)
    assert "Read&quot;]" in diagram
    assert '\nInjected -->' not in diagram
    bullet_skill = SKILL.replace("If the evidence is missing", "- If the evidence is missing")
    assert "- If the evidence is missing" in skill_to_mermaid(bullet_skill)


@pytest.mark.parametrize("active", [
    'flowchart TD\nA[Answer]\nclick A "https://example.com"',
    'flowchart TD\n%%{init: {"securityLevel":"loose"}}%%\nA[Answer]',
    'flowchart TD\nA["<img src=x onerror=alert(1)>"]',
])
def test_active_embedded_mermaid_is_rejected(active):
    with pytest.raises(ValueError, match="active content"):
        skill_to_mermaid(f"```mermaid\n{active}\n```\n")
