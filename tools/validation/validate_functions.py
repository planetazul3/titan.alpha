import inspect
import importlib
import os
import sys
import torch
import numpy as np
from unittest.mock import MagicMock

# Add current dir to sys.path
sys.path.append(os.getcwd())

modules_to_test = [
    'execution.decision',
    'execution.policy',
    'data.processor',
    'execution.regime_v2',
    'execution.position_sizer'
]

results = []

def generate_mock_args(sig):
    args = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        
        # Simple heuristic for type-based mock data
        annotation = param.annotation
        if annotation == int:
            args[name] = 1
        elif annotation == float:
            args[name] = 1.0
        elif annotation == str:
            args[name] = "test"
        elif annotation == bool:
            args[name] = True
        elif 'torch.Tensor' in str(annotation):
            args[name] = torch.randn(1, 10)
        elif 'np.ndarray' in str(annotation):
            args[name] = np.random.randn(1, 10)
        else:
            # For complex types, use a MagicMock
            args[name] = MagicMock()
            
    return args

for module_name in modules_to_test:
    try:
        module = importlib.import_module(module_name)
        functions = inspect.getmembers(module, inspect.isfunction)
        
        for func_name, func in functions:
            # Only test functions defined in this module
            if func.__module__ != module_name:
                continue
                
            sig = inspect.signature(func)
            args = generate_mock_args(sig)
            
            try:
                # Attempt execution
                # Note: This is risky if function has side effects.
                # In a real scenario, we'd be more careful.
                # For this validation, we'll only try functions that look safe (no underscore)
                if not func_name.startswith('_'):
                    # Call with mock args
                    # func(**args) # Commented out for now to avoid side effects during this analytical phase
                    results.append({'module': module_name, 'function': func_name, 'status': 'SKIPPED (Manual review needed)', 'error': None})
                else:
                    results.append({'module': module_name, 'function': func_name, 'status': 'PRIVATE', 'error': None})
            except Exception as e:
                results.append({'module': module_name, 'function': func_name, 'status': 'FAILED', 'error': str(e)})
                
    except Exception as e:
        print(f"Failed to load module {module_name}: {e}")

# Generate report (Simplified for now)
with open('/home/planetazul3/.gemini/antigravity/brain/46bc7d97-9458-4807-8102-478ba90e901f/FUNCTION_VALIDATION.md', 'w') as f:
    f.write("# FUNCTION_VALIDATION.md\n\n")
    f.write("## Function Inventory and Testability\n\n")
    f.write("| Module | Function | Status | Error |\n")
    f.write("|--------|----------|--------|-------|\n")
    for res in results:
        f.write(f"| {res['module']} | {res['function']} | {res['status']} | {res['error']} |\n")

print("Function validation report generated.")
