import nox
from nox.sessions import Session


@nox.session(venv_backend="conda", reuse_venv=True, python="3.8")
def integration(session: Session):
    """Run jupyter notebook test with nbmake.

    How to use nbmake:
    https://semaphoreci.com/blog/test-jupyter-notebooks-with-pytest-and-nbmake

    """
    session.install(
        "pytest",
        "pytest-timeout",
        "pytest-xdist",
        "nbmake",
    )
    if session.posargs:
        posargs = session.posargs
    else:
        posargs = [
            "pai-python-sdk",
        ]

    session.run(
        "pytest",
        "--nbmake",
        "--nbmake-timeout=1800",
        "-n=auto",
        *posargs,
    )
