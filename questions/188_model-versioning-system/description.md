# Model Versioning System

## Problem Statement

Implement a simple model versioning system that tracks model versions, performance metrics, and deployment stages. The system should support basic versioning, model comparison, and simple promotion workflows.

## System Requirements

### Core Components:

1. **Model Registry** - Store and manage model versions
2. **Version Management** - Simple versioning (1.0, 1.1, 1.2, etc.)
3. **Performance Tracking** - Store accuracy and basic metrics
4. **Stage Management** - Track deployment stages (dev, staging, production)

### Key Features:

- **Simple Versioning**: Increment version numbers (1.0, 1.1, 1.2)
- **Performance Tracking**: Store accuracy and basic metrics
- **Stage Management**: Track which stage each version is in
- **Model Comparison**: Compare performance between versions

## Implementation Requirements

### ModelRegistry Class Methods:

1. `register_model(model_name, accuracy, stage="dev")` - Register new model version
2. `get_model(model_name, version=None)` - Get model by version (latest if None)
3. `promote_model(model_name, version, new_stage)` - Move model to new stage
4. `compare_models(model_name, version1, version2)` - Compare two versions
5. `list_models()` - List all registered models

### Model Data Structure:

```python
{
    "model_name": str,
    "version": str,  # e.g., "1.0", "1.1", "1.2"
    "accuracy": float,
    "stage": str,    # "dev", "staging", "production"
    "created_at": str  # ISO timestamp
}
```

### Validation Rules:

- Accuracy must be between 0.0 and 1.0
- Stage must be one of: "dev", "staging", "production"
- Version format: number.number (e.g., "1.0", "1.1")

## Expected Behavior

- First model registration starts at version "1.0"
- Subsequent registrations increment version ("1.1", "1.2", etc.)
- Models can be promoted between stages
- Performance comparison shows accuracy differences
- System handles basic error cases

## Constraints

- Use only standard Python libraries
- Keep it simple and focused on core functionality
- Handle basic edge cases (invalid accuracy, unknown versions)
