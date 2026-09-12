import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run_tests.yml"


def ci_matrix():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["unittest"]["strategy"]["matrix"]


def ci_group_paths():
    """Return the test paths each matrix group runs, one list per group."""
    return {group["name"]: group["paths"].split() for group in ci_matrix()["test-subset"]}


def ci_subsets():
    """Return every (group name, paths) pair the matrix can run, including the `include` additions."""
    matrix = ci_matrix()
    subsets = [(group["name"], group["paths"].split()) for group in matrix["test-subset"]]
    return subsets + [
        (entry["test-subset"]["name"], entry["test-subset"]["paths"].split())
        for entry in matrix.get("include", [])
    ]


def relative(file):
    return str(file.relative_to(REPO_ROOT))


def collect_test_files(path):
    """Expand a group entry into the test files it collects."""
    target = REPO_ROOT / path
    return sorted(target.rglob("test_*.py")) if target.is_dir() else [target]


def test_every_test_file_is_run_by_some_ci_job():
    """A new test file that nobody adds to a group is silently never run. Both the core groups and the
    `include` jobs count here, because a backend's dispatch tests are only reachable from `include` --
    they need an OS or an extra install the core matrix does not give them."""
    covered = {
        file for _, paths in ci_subsets() for path in paths for file in collect_test_files(path)
    }
    on_disk = set((REPO_ROOT / "tests").rglob("test_*.py"))

    missing = sorted(relative(file) for file in on_disk - covered)
    unknown = sorted(relative(file) for file in covered - on_disk)

    assert not missing, f"test files no CI job runs: {missing}"
    assert not unknown, f"CI groups name files that do not exist: {unknown}"


def test_no_test_file_is_in_two_core_groups():
    """The core groups partition `tests/`, so a file in two of them is run twice and the split stops
    balancing. The `include` jobs are deliberately allowed to overlap them -- the Windows smoke job
    re-runs a few files a second time on a second platform."""
    groups_by_file: dict = {}
    for group, paths in ci_group_paths().items():
        for path in paths:
            for file in collect_test_files(path):
                groups_by_file.setdefault(file, []).append(group)

    duplicated = {
        relative(file): groups for file, groups in groups_by_file.items() if len(groups) > 1
    }
    assert not duplicated, f"test files in more than one core CI group: {duplicated}"


def test_every_ci_group_path_exists():
    """The check above only reads the primary matrix, so a typo in the `include` entries — the Windows smoke
    job — would otherwise surface as a file-not-found when that job runs rather than here."""
    missing = [
        f"{group}: {path}"
        for group, paths in ci_subsets()
        for path in paths
        if not (REPO_ROOT / path).exists()
    ]
    assert not missing, f"CI groups name paths that do not exist: {missing}"
