import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool():
    # Setup session and tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Get patterns tool
    patterns_tools = make_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")
    
    # KA #17: FPGrowth (tfidf)
    # catA: alpha (support 3), catB: gamma (support 4)
    res_catA = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})
    res_catB = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})
    
    # Check patterns and support
    # PAMI's getPatterns() returns a dict: {pattern_tuple: support}
    # We need to find the pattern containing "alpha" and check its support
    alpha_pattern = next((p for p in res_catA["patterns"] if "alpha" in p), None)
    assert alpha_pattern is not None
    assert res_catA["patterns"][alpha_pattern] == 3
    
    gamma_pattern = next((p for p in res_catB["patterns"] if "gamma" in p), None)
    assert gamma_pattern is not None
    assert res_catB["patterns"][gamma_pattern] == 4
    
    # KA #20: TopK
    res_topk_A = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 1})
    assert len(res_topk_A["patterns"]) == 1
    
    # KA #20: MaxFPGrowth (should return empty)
    res_max_A = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": 1})
    assert len(res_max_A["patterns"]) == 0
