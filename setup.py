from setuptools import setup, find_packages

setup(
    name="context-engineering-experiments",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tiktoken>=0.5.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
    author="Your Name",
    description="Experimental suite for testing context engineering approaches in LLMs",
    keywords="llm, context, rag, retrieval",
)

