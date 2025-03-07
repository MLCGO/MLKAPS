import pytest
import matplotlib.pyplot as plt


class TestTmpPath:
    def test_validate_tmp_path(self, tmp_path):
        # Use tmp_path for temporary file/directory creation
        temp_dir = tmp_path / "sub"
        temp_dir.mkdir()
        temp_file = temp_dir / "testfile.txt"
        temp_file.write_text("This is a test file.")

        # Add your test logic here
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        fig.savefig(temp_dir / "test.png")

        assert temp_file.read_text() == "This is a test file."
        assert (temp_dir / "test.png").exists()


# Example usage
if __name__ == "__main__":
    pytest.main()
