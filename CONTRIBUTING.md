# Contributing to BSMamba2

Thank you for your interest in contributing to BSMamba2! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/bsmamba2.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests to ensure nothing is broken
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black mypy flake8
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write docstrings in Google style format
- Keep functions focused and modular

### Formatting

Format your code with Black before committing:

```bash
black models/ data/ training/ inference/ utils/
```

### Type Checking

Run mypy to check types:

```bash
mypy models/ data/ training/ inference/ utils/
```

### Linting

Run flake8 to check code quality:

```bash
flake8 models/ data/ training/ inference/ utils/
```

## Testing

Write tests for new features and bug fixes:

```bash
pytest tests/
```

## Documentation

- Update README.md if you add new features
- Add docstrings to all functions and classes
- Update type hints
- Add comments for complex logic

## Commit Messages

Write clear and descriptive commit messages:

- Use present tense: "Add feature" not "Added feature"
- Keep first line under 50 characters
- Add detailed description after blank line if needed

Example:
```
Add support for multi-stem separation

- Implement separate estimation modules for each stem
- Update loss function to handle multiple targets
- Add configuration options for stem selection
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add description of changes to PR
4. Wait for review from maintainers
5. Address any feedback
6. Once approved, PR will be merged

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages and stack traces

## Feature Requests

We welcome feature requests! Please:

- Check if feature already exists or is planned
- Describe the feature clearly
- Explain use case and benefits
- Consider implementation details

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different opinions

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Contact maintainers via email

Thank you for contributing to BSMamba2! 🎵
