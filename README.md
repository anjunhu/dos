# DOS - Deformable Objects Studio
In development.

## Installation
Git modules are used for dependencies. To clone the repository with all submodules, use the following command:
```
git clone --recursive
```

## Development

### TODO
- [ ] Modular checkpointing: Save models as a dictionary of its networks for better compatibility between different versions of the code.

### Style guide
Use Black for code formatting.

Automatically sort imports with isort.

Prefer einops for tensor operations due to its simplicity and readability.

### Problems and questions
Mesh rendering uses vertex tangents but we do not use UV textures. Solution?
