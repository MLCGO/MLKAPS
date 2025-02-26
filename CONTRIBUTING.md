# Contributing Rules
* The main branch is for pre-release code. Only selected users can create pull requests (PRs) and merge into this branch. In principle, this is only to push development changes to the main branch or to fix critical issues in released code.
  * If you find such critical issues not related to security, please promptly open an issue with the appropriate label and reference @titeup in the issue.
  * If you find an issue related to security, please refer to the [security](SECURITY.md) guidelines.
* The dev branch is the current development branch. Any user can propose a PR to this branch.
  * All contributor PRs should be made to the dev branch.
  * After code freeze, we will block PR merges to the dev branch until all extensive tests are successfully completed and changes are pushed to the main branch.
* All contributions must provide:
  * Sphinx documentation for new functions and modules.
  * Author(s) must provide a  tests covering all  new features.
  * test time must be kept small, if possible reuse and extend existing test
* To get a PR accepted:
  * PR description must describe new contributions, bug fixes, and enhancements. In addition, it can describe additional and/or missing tasks to be completed to encourage others to contribute to your PR.
  * Ask 2 reviewers when ready. If you do not know who to assign, assign @titeup and comment that you would like to get your PR reviewed.
  * Code must be tested before committing. If the CI is unavailable, it is the reviewer's responsibility to run the tests.
  * New external dependencies must be clearly documented and justified in a separate PR comment.
* For license and copyright, please refer to [Licence file](LICENSE). All contributions to this mlkaps repository must be compatible with the said license. It is the author's first responsibility to ensure that the proposed contribution is compatible and does not violate any rules.

# Coding Rules
* This project is primarily in Python; we prefer to avoid other scripting languages or compiled code as much as possible.
* Coding style:
  * All Python code must adhere to PEP 8 guidelines.
    * Use `pylint` to check for linting issues before submitting a PR.
    * Use `black` to format your code before submitting a PR.
  * All C and C++ code must adhere to the project's coding standards.
    * Use `clang-tidy` and `clang-format` to check for linting issues and format your code before submitting a PR.