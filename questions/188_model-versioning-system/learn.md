# Model Versioning System - Learning Guide

## Overview

This problem introduces you to MLOps fundamentals through a simplified model versioning system. You'll learn core concepts of model lifecycle management, version tracking, and deployment coordination without overwhelming complexity.

## Key Concepts

### 1. Model Versioning
- **Simple Versioning**: Use incrementing version numbers (1.0, 1.1, 1.2)
- **Automatic Increment**: Each new registration increments the version
- **Version Tracking**: Keep history of all model versions

### 2. Model Registry Design
- **Centralized Storage**: Store all model information in one place
- **Basic Metadata**: Track essential information (name, version, accuracy, stage)
- **Simple Queries**: Easy retrieval and comparison of models

### 3. Deployment Stages
```
Development → Staging → Production
     ↓           ↓          ↓
   Testing   Validation   Live Use
```

### 4. Core Data Structure
```python
{
    "model_name": "sentiment_classifier",
    "version": "1.0",
    "accuracy": 0.95,
    "stage": "dev",
    "created_at": "2024-01-15T10:30:00Z"
}
```

## Implementation Strategies

### 1. Data Storage
```python
# Simple approach using nested dictionaries
self.models = {
    "model_name": {
        "1.0": {"accuracy": 0.95, "stage": "dev", "created_at": "..."},
        "1.1": {"accuracy": 0.97, "stage": "dev", "created_at": "..."}
    }
}
```

### 2. Version Management
- Start with version "1.0" for first registration
- Increment minor version for subsequent registrations
- Parse version strings to determine next version

### 3. Stage Promotion
- Validate that model exists before promotion
- Update stage information for specific version
- Return success/failure status

### 4. Model Comparison
- Retrieve both model versions
- Calculate accuracy difference
- Return comparison results

## Common Implementation Patterns

### 1. Version Increment Logic
```python
def _get_next_version(self, model_name):
    if model_name not in self.models:
        return "1.0"
    
    versions = list(self.models[model_name].keys())
    latest_version = max(versions, key=lambda v: float(v))
    major, minor = map(int, latest_version.split('.'))
    return f"{major}.{minor + 1}"
```

### 2. Validation
```python
def _validate_inputs(self, model_name, accuracy, stage):
    if not model_name or not isinstance(model_name, str):
        return False, "Invalid model name"
    if not 0.0 <= accuracy <= 1.0:
        return False, "Accuracy must be between 0.0 and 1.0"
    if stage not in ["dev", "staging", "production"]:
        return False, "Invalid stage"
    return True, ""
```

## Key Learning Points

### 1. System Design Basics
- **Separation of Concerns**: Each method has a single responsibility
- **Data Integrity**: Validate inputs and handle edge cases
- **Error Handling**: Return meaningful results for success/failure

### 2. MLOps Concepts
- **Model Lifecycle**: Track models from development to production
- **Version Control**: Maintain history of model improvements
- **Stage Management**: Control deployment progression
- **Performance Tracking**: Monitor model quality over time

### 3. Real-world Relevance
This simplified system mirrors concepts used in:
- **MLflow Model Registry**: Industry-standard model management
- **AWS SageMaker**: Cloud-based ML model deployment
- **Azure ML**: Microsoft's ML platform
- **Kubeflow**: Kubernetes-based ML workflows

## Common Pitfalls to Avoid

1. **Version Parsing**: Handle version string format correctly
2. **Edge Cases**: Check for non-existent models/versions
3. **Data Validation**: Ensure accuracy is within valid range
4. **Stage Validation**: Only allow valid stage transitions
5. **Return Types**: Match expected return formats

## Testing Approach

- **Happy Path**: Test normal operations
- **Edge Cases**: Test with invalid inputs
- **Error Handling**: Verify proper error responses
- **State Management**: Ensure data consistency

This problem provides a solid foundation for understanding MLOps system design while remaining approachable for learning the core concepts.
