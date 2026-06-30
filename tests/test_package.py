import auralbench


def test_version_contract():
    assert auralbench.__version__ == "0.1.2"
    assert auralbench.__all__ == [
        "__version__",
        "evaluate_dataset",
        "report_csv",
        "report_html",
        "report_json",
        "character_error_rate",
        "word_error_rate",
    ]
