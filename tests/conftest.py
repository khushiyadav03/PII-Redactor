import pytest

def pytest_sessionstart(session):
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError) as exc:
        pytest.exit(
            "\n======================================================================\n"
            "CRITICAL SETUP ERROR: spaCy model 'en_core_web_sm' is not installed.\n"
            "Please run: python -m spacy download en_core_web_sm\n"
            "before running tests or the application.\n"
            "======================================================================\n"
        )
