import nox
from nox.sessions import Session


PASSING_ENVIRONMENTS = {
    "PAI_TEST_CONFIG": "test.ini",
}


@nox.session(reuse_venv=True)
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
        "--timeout",
        "3000",
        "--nbmake",
        "-n=auto",
        *posargs,
    )
