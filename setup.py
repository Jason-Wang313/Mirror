from setuptools import setup, find_packages

setup(
    name="mirror-bench",
    version="0.1.0",
    description="Project MIRROR - Measuring metacognitive capacity in LLMs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "google-genai>=1.0.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.0.0",
        "tiktoken>=0.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "jsonlines>=4.0.0",
    ],
)
