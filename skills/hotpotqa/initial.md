# HotpotQA Initial Skill

## Global Guidance
This neutral initial skill is a local HotpotQA adapter, not a pretrained or evolved result.
Use only the supplied passages. Resolve references across passages when needed and keep the answer concise.
Do not use reference answers or supporting-fact annotations as model inputs.

## Node Lists

### Read the question and passages
Identify the requested entity, relation, or comparison. Read the supplied passage titles and sentences.

### Connect relevant evidence
For a multi-hop question, connect the relevant entities and facts across passages.
For a comparison, compare the attributes explicitly requested by the question.
If evidence is insufficient, do not invent a supporting fact.

### Return the final answer
Return the minimal answer inside <answer>...</answer> tags.
For a yes/no question, return yes or no when supported by the evidence.

## Task Graphs

### Question answering
**Use when:** A HotpotQA question and its context passages are supplied.
**Workflow:**
1. Read the question and passages
2. Connect relevant evidence
3. Return the final answer
