from setuptools import setup, find_packages

setup(
    name="fastreader",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain",
        "langgraph",
        "langchain_ollama",
        "langchain_openai",
        "langchain_community",
        "ruff",
        "youtube_transcript_api",
    ],
    entry_points={
        "console_scripts": [
            "fastreader=app.main:run",
        ],
    },
)
