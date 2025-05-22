import pytest
import matplotlib.pyplot as plt
import os


class TestTmpPath:
    def test_validate_tmp_path(self, tmp_path):
        # Use tmp_path for temporary file/directory creation
        temp_dir = tmp_path / "sub"
        temp_dir.mkdir()
        temp_file = temp_dir / "testfile.txt"
        temp_file.write_text("This is a test file.")

        # Windows has a bug with Python 3.13 in loading tk/tcl into a virtual environment.
        # https://github.com/python/cpython/issues/125235.  This causes the plot routine to fail.
        if os.name == "nt":
            pass
        else:
            # Add your test logic here
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            fig.savefig(temp_dir / "test.png")
            assert (temp_dir / "test.png").exists()


# Example usage
if __name__ == "__main__":
    pytest.main()
