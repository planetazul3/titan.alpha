
import ast
import os
from pathlib import Path
import networkx as nx

def analyze_dependencies(root_dir: str):
    root = Path(root_dir)
    graph = nx.DiGraph()
    
    file_map = {}
    
    # 1. Map all python files
    for path in root.rglob("*.py"):
        if "venv" in str(path) or ".git" in str(path):
            continue
        module_name = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        file_map[module_name] = path
        graph.add_node(module_name)

    # 2. Parse imports
    for module, path in file_map.items():
        try:
            with open(path, "r") as f:
                tree = ast.parse(f.read(), filename=str(path))
            
            for node in ast.walk(tree):
                target = None
                if isinstance(node, ast.Import):
                    for name in node.names:
                        target = name.name
                        if target in file_map:
                            graph.add_edge(module, target)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Absolute or relative import
                        if node.level == 0:
                            target = node.module
                        else:
                            # Resolve relative import (simplified)
                            parts = module.split(".")[:-node.level]
                            if node.module:
                                parts.append(node.module)
                            target = ".".join(parts)
                        
                        if target in file_map:
                            graph.add_edge(module, target)
                            
        except Exception as e:
            print(f"Error parsing {module}: {e}")

    # 3. Find Cycles
    cycles = list(nx.simple_cycles(graph))
    if cycles:
        print(f"\nFound {len(cycles)} circular dependencies:")
        for cycle in cycles:
            print(f"  {' -> '.join(cycle)} -> {cycle[0]}")
    else:
        print("\nNo circular dependencies found.")
        
    return graph

if __name__ == "__main__":
    print("Analyzing x.titan dependencies...")
    try:
        import networkx
    except ImportError:
        print("Please install networkx: pip install networkx")
        exit(1)
        
    g = analyze_dependencies(".")
