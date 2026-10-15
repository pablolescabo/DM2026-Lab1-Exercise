import pytest
import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")
    result = check_missing_tool.invoke({})
    
    assert result["missing_count"] == 1

def test_check_duplicates_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    result_no_drop = check_duplicates_tool.invoke({"drop": False})
    assert result_no_drop["duplicate_count"] == 1
    assert result_no_drop["remaining_rows"] == 8
    
    result_drop = check_duplicates_tool.invoke({"drop": True})
    assert result_drop["remaining_rows"] == 6

def test_tokenize_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    tokenize_tool.invoke({})
    
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.dataframe["unigrams"].apply(len).sum() == 27

def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

def test_sample_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    
    assert result["picked_indices"] == [1, 5, 0, 7]
    assert result["remaining_rows"] == 4

def test_describe_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv", 
        "text_column": "text", 
        "label_column": "label"
    })
    
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")
    result = describe_data_tool.invoke({})
    
    assert session.dataframe["text_length"].tolist() == [29, 23, 17, 30, 24, 18, 18, 0]
    
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], 0.001) == 19.875
    assert pytest.approx(overall["std"], 0.001) == 9.4330
    assert overall["min"] == 0
    assert pytest.approx(overall["25%"], 0.001) == 17.75
    assert pytest.approx(overall["50%"], 0.001) == 20.5
    assert pytest.approx(overall["75%"], 0.001) == 25.25
    assert overall["max"] == 30
    
    cat_stats = {row["category_name"]: row for row in result["category_stats"]}
    
    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert pytest.approx(catA["mean"], 0.001) == 17.25
    assert pytest.approx(catA["std"], 0.001) == 12.5
    assert catA["min"] == 0
    assert pytest.approx(catA["25%"], 0.001) == 12.75
    assert pytest.approx(catA["50%"], 0.001) == 20.0
    assert pytest.approx(catA["75%"], 0.001) == 24.5
    assert catA["max"] == 29
    
    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert pytest.approx(catB["mean"], 0.001) == 22.5
    assert pytest.approx(catB["std"], 0.001) == 5.7446
    assert catB["min"] == 18
    assert pytest.approx(catB["25%"], 0.001) == 18.0
    assert pytest.approx(catB["50%"], 0.001) == 21.0
    assert pytest.approx(catB["75%"], 0.001) == 25.5
    assert catB["max"] == 30
    
    assert session.pending_figure is not None

def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    # Test normal inspection
    result = inspect_data_tool.invoke({"n": 2, "max_chars": 10})
    assert result["shape"] == [8, 3] # 8 rows, 3 columns (text, category_name, category)
    assert len(result["head"]) == 2
    assert result["head"][0]["text"].endswith("...")
    
    # Test invalid inputs
    assert "error" in inspect_data_tool.invoke({"n": 0})
    assert "error" in inspect_data_tool.invoke({"max_chars": 0})
