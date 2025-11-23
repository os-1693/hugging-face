import os

import nbformat

src_dir = "src/"
output_dir = "notebooks/"

files = [f for f in os.listdir(src_dir) if f.endswith(".py")]

for file in files:
    with open(os.path.join(src_dir, file), "r", encoding="utf-8") as f:
        content = f.read()

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(content))
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.8.5"},
        "colab": {"provenance": []},
    }

    output_name = file.replace(".py", "_colab.ipynb")
    with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

print("Conversion completed.")
