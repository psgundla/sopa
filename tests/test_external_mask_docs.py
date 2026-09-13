import json
from pathlib import Path

import numpy as np
import shapely
from spatialdata import transform


def test_external_mask_example():
    """Run the documented synthetic example and independently check its counts."""
    notebook = Path(__file__).parents[1] / "docs/tutorials/custom_segmentation.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    snippets = [
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "external-mask" in cell["metadata"].get("tags", [])
    ]
    assert len(snippets) == 2
    namespace = {}
    exec(compile(snippets[0], str(notebook), "exec"), namespace)  # noqa: S102 -- execute tracked documentation only
    sdata = namespace["external_sdata"]
    mask_before = namespace["mask"].copy()
    original_cells = sdata["cells"].copy(deep=True)
    transcripts_before = sdata["transcripts"].compute().copy(deep=True)
    exec(compile(snippets[1], str(notebook), "exec"), namespace)  # noqa: S102 -- execute tracked documentation only

    np.testing.assert_array_equal(namespace["mask"], mask_before)
    assert sdata["cells"].equals(original_cells)
    assert sdata["transcripts"].compute().equals(transcripts_before)
    table = sdata["external_table"]
    # Toy transcripts have their own affine: compare in the image's pixel frame.
    aligned = transform(sdata["transcripts"], to_coordinate_system="global").compute()
    points = shapely.points(aligned.x, aligned.y)
    expected = np.zeros(table.shape, dtype=int)
    for row, polygon in enumerate(sdata["external_boundaries"].geometry):
        inside = shapely.contains(polygon, points)
        for col, gene in enumerate(table.var_names):
            expected[row, col] = (inside & (aligned["genes"] == gene)).sum()
    np.testing.assert_array_equal(table.X.toarray(), expected)
