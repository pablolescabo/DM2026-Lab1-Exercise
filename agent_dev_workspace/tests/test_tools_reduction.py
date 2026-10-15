import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools
from tools_dtm import make_tools as make_dtm_tools

def test_reduce_dimensions_tool():
    # Setup session and tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Build DTM (required for reduction)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get reduction tool
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # KA #7: PCA
    res_pca = reduce_dimensions_tool.invoke({"method": "pca"})
    assert session.pending_figure is not None
    session.pending_figure = None # Clear for next test
    expected_pca = [
        [2.3669, 0.8871], [1.7039, 0.2642], [1.0408, -0.3587],
        [-2.0759, 1.0129], [-1.4554, 0.3405], [-0.8349, -0.3319],
        [-0.8349, -0.3319], [0.0894, -1.4822]
    ]
    assert np.allclose(res_pca["coordinates"], expected_pca, atol=1e-3)
    assert np.allclose(res_pca["explained_variance_ratio"], [0.7532, 0.1965], atol=1e-3)
    
    # KA #8: t-SNE (check shape only due to stochastic nature)
    res_tsne = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    assert session.pending_figure is not None
    session.pending_figure = None # Clear for next test
    coords_tsne = np.array(res_tsne["coordinates"])
    assert coords_tsne.shape == (8, 2)
    
    # KA #9: UMAP
    res_umap = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert session.pending_figure is not None
    session.pending_figure = None # Clear for next test
    expected_umap = [
        [6.6811, -6.8218], [6.9907, -7.2245], [7.7022, -7.3712],
        [10.2228, -6.2310], [9.8759, -5.6537], [9.3831, -6.5748],
        [9.2153, -5.8297], [8.5860, -7.2622]
    ]
    assert np.allclose(res_umap["coordinates"], expected_umap, atol=1e-3)

def test_binarize_labels_tool():
    # Setup session and tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Set labels
    session.set_labels(session.dataframe["category_name"].values)
    
    # Get binarize labels tool
    reduction_tools = make_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    # Run tool
    res = binarize_labels_tool.invoke({})
    
    # Verify results
    binarized = session.artifacts["binarized_labels"]
    assert binarized.shape == (8, 2)
    assert set(binarized.columns) == {"catA", "catB"}
    assert binarized.sum().sum() == 8
    assert all(binarized.isin([0, 1]).all())
