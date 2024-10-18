# 🦜️🔗 LangChain {partner}

This repository contains 1 package with {partner} integrations with LangChain:

- [langchain-{package_lower}](https://pypi.org/project/langchain-{package_lower}/)

## Initial Repo Checklist (Remove this section after completing)

This setup assumes that the partner package is already split. For those instructions,
see [these docs](https://python.langchain.com/docs/contributing/integrations#partner-packages).

Code (auto ecli)

- [ ] Fill out the readme above (for folks that follow pypi link)
- [ ] Copy package into /libs folder
- [ ] Update these fields in /libs/*/pyproject.toml

    - `tool.poetry.repository`
    - `tool.poetry.urls["Source Code"]`

Workflow code (auto ecli)

- [ ] Populate .github/workflows/_release.yml with `on.workflow_dispatch.inputs.working-directory.default`
- [ ] Configure `LIB_DIRS` in .github/scripts/check_diff.py

Workflow code (manual)

- [ ] Add secrets as env vars in .github/workflows/_release.yml

In github (manual)

- [ ] Add integration testing secrets in Github (ask Erick for help)
- [ ] Add partner collaborators in Github (ask Erick for help)
- [ ] "Allow auto-merge" in General Settings 
- [ ] Only "Allow squash merging" in General Settings
- [ ] Set up ruleset matching CI build (ask Erick for help)
    - name: ci build
    - enforcement: active
    - bypass: write
    - target: default branch
    - rules: restrict deletions, require status checks ("CI Success"), block force pushes
- [ ] Set up ruleset
    - name: require prs
    - enforcement: active
    - bypass: none
    - target: default branch
    - rules: restrict deletions, require a pull request before merging (0 approvals, no boxes), block force pushes

Pypi (manual)

- [ ] Add new repo to test-pypi and pypi trusted publishing (ask Erick for help)

Slack

- [ ] Set up release alerting in Slack (ask Erick for help)

release:
/github subscribe langchain-ai/langchain-{partner_lower} releases workflows:{name:"release"}
/github unsubscribe langchain-ai/langchain-{partner_lower} issues pulls commits deployments
