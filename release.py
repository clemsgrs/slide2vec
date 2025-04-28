import subprocess
import re


def run(cmd: str, check: bool = True) -> str:
    result = subprocess.run(cmd, shell=True, check=check, stdout=subprocess.PIPE)
    return result.stdout.decode().strip()


def get_current_version() -> str:
    out = run("bumpver show")
    for line in out.splitlines():
        if line.startswith("Current Version"):
            _, version = line.split(":")
            return version.strip().strip('"').strip("'")
    raise RuntimeError("Could not find current_version")


def bump_version(level: str = "patch") -> str:
    print(f"🔧 Bumping version with level: {level}")
    run(f"bumpver update --{level} --commit")
    return get_current_version()


def push_tag_and_branch(version: str) -> str:
    branch = f"release-v{version}"
    print(f"🌿 Creating branch {branch}...")
    run(f"git checkout -b {branch}")
    run(f"git push origin {branch}")
    tag = f"v{version}"
    run(f"git push origin {tag}")
    return branch


def create_pull_request(branch: str, version: str) -> None:
    print(f"🔁 Creating pull request for {branch} → main...")
    run(
        f'gh pr create --title "Release v{version}" '
        f'--body "This PR bumps the version to v{version} and tags the release." '
        f"--base main --head {branch}"
    )


def open_release_draft(tag: str) -> None:
    repo = run("git remote get-url origin")
    match = re.search(r"github\.com[:/](.*?)(\.git)?$", repo)
    if not match:
        print("❌ Could not detect GitHub repo")
        return
    repo_path = match.group(1)
    url = f"https://github.com/{repo_path}/releases/new?tag={tag}&title={tag}"
    print(f"🌐 Open the release page:\n{url}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=["patch", "minor", "major"],
        default="patch",
        help="Version bump level",
    )
    parser.add_argument(
        "--no-draft", action="store_true", help="Don't open GitHub release page"
    )
    parser.add_argument(
        "--no-pr", action="store_true", help="Don't create pull request"
    )
    args = parser.parse_args()

    version = bump_version(args.level)
    branch = push_tag_and_branch(version)

    if not args.no_pr:
        create_pull_request(branch, version)

    if not args.no_draft:
        open_release_draft(f"v{version}")
