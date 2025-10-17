# Contributing to Optical-Guided-Super-Resolution

Thank you for considering contributing to this project! Your help in enhancing optical-guided super-resolution techniques is greatly appreciated. We welcome contributions of all sizes, from bug fixes to new features like additional backbones or dataset integrations.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you agree to uphold it. Please report unacceptable behavior to [manasmehta1123@example.com](mailto:your-email@example.com).

## Getting Started

1. **Fork the repo** and clone your fork.
2. **Install dependencies**: Run `pip install -r requirements.txt` (or use the provided Dockerfile).
3. **Create a branch** for your feature: `git checkout -b feature/amazing-new-model`.
4. **Test locally**: Ensure your changes pass any existing tests (e.g., `pytest`).
5. **Commit and push**: Use descriptive messages (e.g., "Add SAR-optical alignment module").

## How to Contribute

- **Bug Reports**: Open an issue with a clear title, steps to reproduce, and expected vs. actual behavior. Include environment details (e.g., PyTorch version, GPU).
- **Feature Requests**: Suggest ideas in issues—label them "enhancement" if you plan to implement.
- **Pull Requests (PRs)**:
  - Reference the related issue (e.g., "Fixes #42").
  - Keep PRs focused on one change.
  - Update docs/README if your code affects usage.
  - Ensure code style: Run `black .` for formatting and `flake8` for linting.
- **New Models/Datasets**: Add them modularly in `src/models/` or `data/`. Include benchmarks vs. baselines.

## Development Workflow

- Use GitHub Issues for tracking.
- For experiments, log with Weights & Biases (optional—integrate via `wandb.init()`).
- Before merging: CI will run tests via GitHub Actions.

## Questions?

Ask in an issue or reach out via email.

Happy contributing! 🚀
