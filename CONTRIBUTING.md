# Contributing to FDFI

Thank you for your interest in contributing to FDFI!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/FDFI.git
   cd FDFI
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```

3. Format your code:
   ```bash
   make format
   ```

4. Run linters:
   ```bash
   make lint
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add your descriptive commit message"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Add type hints where appropriate
- Write docstrings for all public functions and classes

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Use pytest for testing

## Documentation

- Update documentation for new features
- Include docstrings with examples
- Update README.md if needed

Build the documentation locally:

```bash
cd docs
python -m sphinx -b html . _build/html
open _build/html/index.html
```

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Keep PRs focused and atomic

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing!
