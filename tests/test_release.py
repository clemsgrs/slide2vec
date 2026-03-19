import release


def test_push_branch_and_tag_uses_plain_version_tags(monkeypatch, capsys):
    commands: list[str] = []

    def fake_run(cmd: str, check: bool = True) -> str:
        commands.append(cmd)
        return ""

    monkeypatch.setattr(release, "run", fake_run)

    release.push_branch_and_tag("release-2.0.3", "2.0.3")

    assert commands == [
        "git push origin release-2.0.3",
        "git tag 2.0.3",
        "git push origin 2.0.3",
    ]
    assert "Creating and pushing tag 2.0.3" in capsys.readouterr().out


def test_push_tag_and_branch_uses_plain_version_names(monkeypatch, capsys):
    commands: list[str] = []

    def fake_run(cmd: str, check: bool = True) -> str:
        commands.append(cmd)
        if cmd == "git tag":
            return ""
        return ""

    monkeypatch.setattr(release, "run", fake_run)

    branch = release.push_tag_and_branch("2.0.3")

    assert branch == "release-2.0.3"
    assert commands == [
        "git checkout -b release-2.0.3",
        "git push origin release-2.0.3",
        "git tag",
        "git tag 2.0.3",
        "git push origin 2.0.3",
    ]
    output = capsys.readouterr().out
    assert "Creating branch release-2.0.3" in output
    assert "Creating tag 2.0.3" in output


def test_create_pull_request_and_release_draft_use_plain_versions(monkeypatch, capsys):
    commands: list[str] = []

    def fake_run(cmd: str, check: bool = True) -> str:
        commands.append(cmd)
        if cmd == "git remote get-url origin":
            return "git@github.com:example/slide2vec.git"
        return ""

    monkeypatch.setattr(release, "run", fake_run)

    release.create_pull_request("release-2.0.3", "2.0.3")
    release.open_release_draft("2.0.3")

    assert commands == [
        'gh pr create --title "Release 2.0.3" --body "This PR bumps the version to 2.0.3 and tags the release." --base main --head release-2.0.3',
        "git remote get-url origin",
    ]
    assert "releases/new?tag=2.0.3&title=2.0.3" in capsys.readouterr().out
