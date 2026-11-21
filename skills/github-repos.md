# GitHub repositories (Back-only github-mcp)

GitHub API access lives on the Back agents (back-1..back-5) via
`mcp_call(server='github-mcp', tool=..., arguments=...)`. Front does not hold
GitHub credentials: when the owner asks about repositories, Front says
"Checking ..." and delegates with `delegate_task(brief, role='back')`.

## Reading any public repository

Use `repo_overview` for a one-call read (metadata + README + top-level files):

```
mcp_call(server='github-mcp', tool='repo_overview',
         arguments={"repo": "https://github.com/roatienza/deep-learning-experiments"})
```

Known course repository: **roatienza/deep-learning-experiments** — Prof. Rowel
Atienza's deep learning experiments (AI 231 / advanced deep learning course
material). Read its README and notebooks with `repo_overview`, then `get_file`
for individual paths, e.g. `{"repo": "roatienza/deep-learning-experiments",
"path": "README.md"}`. Use `list_commits` for recent activity.

For deeper reads: `get_file` (file content or directory listing),
`search_code` (query like `"repo:roatienza/deep-learning-experiments transformer"`).

## Creating / editing repositories (demo flow)

1. `create_repo` with `{"name": "...", "private": false, "auto_init": true}`
   creates a **public** repo with a README on `main`.
2. `create_or_update_file` writes/overwrites a file (auto-resolves sha).
3. `edit_file` replaces `old_text` with `new_text` and commits.
4. `list_commits`, `create_branch`, `create_issue`, `create_pull_request`
   cover the rest. Always return the exact `html_url` to the owner.

Editing someone else's public repo directly requires write permission on that
repo; otherwise fork-style flow: read with `repo_overview`/`get_file`, then
create the owner's own public copy with `create_repo` + `create_or_update_file`.
