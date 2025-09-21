# Contributing to PillPilot

Thank you for your interest in contributing to PillPilot! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Search through closed issues for similar problems
3. Provide detailed information about the problem

**Issue Template:**
```markdown
**Bug Description**
A clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
- OS: [e.g. Windows 10, macOS 12, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Browser: [e.g. Chrome 95, Firefox 94]

**Additional Context**
Add any other context about the problem here.
```

### Suggesting Features

We welcome feature suggestions! Please:
1. Check if the feature has been requested before
2. Provide a clear description of the feature
3. Explain why it would be useful
4. Consider implementation complexity

**Feature Request Template:**
```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
Describe your proposed solution.

**Alternatives**
Describe any alternative solutions you've considered.

**Additional Context**
Add any other context or screenshots about the feature request.
```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- pip

### Setup Steps

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/pillpilot.git
   cd pillpilot
   ```

2. **Create a development branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```

## 📝 Coding Standards

### Python Code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

**Example:**
```python
def calculate_stockout_risk(quantity: int, daily_consumption: float) -> str:
    """
    Calculate stockout risk level based on current quantity and consumption.
    
    Args:
        quantity: Current stock quantity
        daily_consumption: Average daily consumption rate
        
    Returns:
        Risk level as string ('Low', 'Medium', 'High', 'Critical')
    """
    if daily_consumption <= 0:
        return 'Low'
    
    days_of_stock = quantity / daily_consumption
    
    if days_of_stock <= 7:
        return 'Critical'
    elif days_of_stock <= 14:
        return 'High'
    elif days_of_stock <= 30:
        return 'Medium'
    else:
        return 'Low'
```

### JavaScript Code
- Use ES6+ features
- Follow consistent naming conventions (camelCase)
- Add comments for complex logic
- Use async/await for API calls

**Example:**
```javascript
async function loadChartData(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error loading chart data:', error);
        throw error;
    }
}
```

### HTML/CSS
- Use semantic HTML elements
- Follow BEM methodology for CSS classes
- Keep CSS organized and commented
- Use consistent indentation (2 spaces)

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_api.py

# Run with coverage
python -m pytest --cov=app tests/
```

### Writing Tests
- Write tests for new features
- Aim for good test coverage
- Test both success and error cases
- Use descriptive test names

**Example:**
```python
def test_calculate_stockout_risk():
    """Test stockout risk calculation with various inputs."""
    # Test critical risk
    assert calculate_stockout_risk(10, 2) == 'Critical'
    
    # Test low risk
    assert calculate_stockout_risk(100, 1) == 'Low'
    
    # Test edge case
    assert calculate_stockout_risk(0, 1) == 'Critical'
```

## 📋 Pull Request Process

### Before Submitting
1. **Test your changes**
   - Run the application locally
   - Test all affected functionality
   - Check for any console errors

2. **Update documentation**
   - Update README.md if needed
   - Add/update docstrings
   - Update API documentation

3. **Check code quality**
   - Run linting tools
   - Ensure code follows style guidelines
   - Remove debug code and comments

### PR Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process
1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Approval** from at least one maintainer

## 🏗️ Project Structure

Understanding the codebase:

```
pillpilot/
├── app.py                          # Main Flask application
├── demand_forecasting_model.py     # ML models and algorithms
├── templates/                      # HTML templates
│   ├── index.html                 # Main dashboard
│   ├── transfer.html              # Transfer suggestions
│   └── analytics.html             # Analytics dashboard
├── static/                        # Static assets
│   ├── style.css                  # Main stylesheet
│   └── script.js                  # Frontend JavaScript
├── tests/                         # Test files
├── docs/                          # Documentation
└── requirements.txt               # Python dependencies
```

## 🎯 Areas for Contribution

### High Priority
- **Performance optimization** for large datasets
- **Additional ML models** for demand forecasting
- **Mobile responsiveness** improvements
- **API documentation** with Swagger/OpenAPI

### Medium Priority
- **Unit tests** for existing functionality
- **Integration tests** for API endpoints
- **Error handling** improvements
- **Logging** and monitoring

### Low Priority
- **UI/UX enhancements**
- **Additional chart types**
- **Export functionality**
- **Multi-language support**

## 🐛 Debugging Tips

### Common Issues
1. **Import errors**: Check virtual environment activation
2. **Port conflicts**: Use different port or kill existing process
3. **CSV format issues**: Verify column names and data types
4. **Chart loading errors**: Check browser console for JavaScript errors

### Debug Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
python app.py
```

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Ask for help in pull request comments

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub** contributor statistics

Thank you for contributing to PillPilot! 🚀

