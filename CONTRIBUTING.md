# Contributing guidelines

## Branches

* The `main` branch is for pre-release code. Only selected users can create pull requests (PRs) and merge into this branch. In principle, this is only to merge changes into the main branch or to fix critical issues in released code.
  * If you find such critical issues not related to security, please promptly open an issue with the appropriate label and reference @titeup in the issue.
  * If you find an issue related to security, please refer to the [security](SECURITY.md) guidelines.
* The `dev` branch is the current development branch. Any user can propose a PR to this branch.
  * All contributor PRs should be made against the `dev` branch.
  * After code freeze, we will block PR merges to the `dev` branch until all extensive tests are successfully completed and changes are pushed to the `main` branch.

## Pull Requests

* All contributions must provide:
  * Sphinx documentation for new functions and modules.
  * Tests covering all new features or bug fixes. Integration tests can be expensive, prefer unit tests or minimal test cases.
* To facilitate PR acceptance, please follow these guidelines:
  * PR description must describe new contributions, bug fixes, and enhancements. In addition, they can describe followup / optional tasks to be completed to encourage others to contribute to your PR.
  * Ask two reviewers when ready. If you do not know who to assign, assign @titeup and comment that you would like to get your PR reviewed.
  * Code must be tested before committing. If the CI is unavailable, the reviewer will manually run the test suite.
  * New external dependencies must be clearly documented and justified in a standalone PR comment.
* For license and copyright, please refer to [Licence file](LICENSE). All contributions to this MLKAPS repository must be compatible with the said license. It is the author's first responsibility to ensure that the proposed contribution does not violate any rules.

## Coding Rules

* This project is written primarily in Python; other script languages and compiled code should be avoided as much as possible.
* Coding style:
  * All Python code must adhere to PEP 8 guidelines.
    * Use `flake8` to check for linting issues before submitting a PR.
    * Use `black` to format your code before submitting a PR.
  * All C and C++ code must adhere to the project's coding standards.
    * Use `clang-tidy` and `clang-format` to check for linting issues and format your code before submitting a PR.
* All code must be well-documented using Sphinx.
