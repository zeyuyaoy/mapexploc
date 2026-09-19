"""Disposable hosted CI probe; this branch must never be merged."""


def test_required_python_check_rejects_failure():
    assert False, "Intentional CI enforcement probe"
