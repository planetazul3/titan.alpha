import os
import re
import json

def get_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        if 'venv' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def parse_imports(file_path):
    imports = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Match: import x.y.z
            match1 = re.match(r'^import\s+([\w\.]+)', line)
            if match1:
                imports.append(match1.group(1))
            # Match: from x.y import z
            match2 = re.match(r'^from\s+([\w\.]+)\s+import', line)
            if match2:
                imports.append(match2.group(1))
    return imports

def build_dependency_graph(files):
    graph = {}
    for file in files:
        module_name = file.replace('./', '').replace('/', '.').replace('.py', '')
        # Special case for __init__.py
        if module_name.endswith('.__init__'):
            module_name = module_name[:-9]
        
        imports = parse_imports(file)
        # Filter for internal imports only
        internal_imports = []
        for imp in imports:
            # Check if any part of the import matches our internal folders
            parts = imp.split('.')
            if parts[0] in ['execution', 'models', 'data', 'training', 'core', 'observability', 'api', 'utils', 'config', 'scripts']:
                internal_imports.append(imp)
        
        graph[module_name] = internal_imports
    return graph

def find_circular_dependencies(graph):
    circular = []
    
    def visit(node, path):
        if node in path:
            cycle = path[path.index(node):] + [node]
            circular.append(cycle)
            return
        
        if node not in graph:
            return
            
        for neighbor in graph[node]:
            visit(neighbor, path + [node])

    # This is a bit naive but works for small graphs
    for node in graph:
        visit(node, [])
        
    # Deduplicate cycles
    unique_cycles = []
    seen = set()
    for cycle in circular:
        # Normalize cycle by sorting or finding minimum rotation
        sorted_cycle = tuple(sorted(cycle[:-1])) # Exclude tail duplicate
        if sorted_cycle not in seen:
            unique_cycles.append(cycle)
            seen.add(sorted_cycle)
            
    return unique_cycles

if __name__ == "__main__":
    files = get_python_files('.')
    graph = build_dependency_graph(files)
    
    cycles = find_circular_dependencies(graph)
    
    print("# DEPENDENCY_MAP.md")
    print("\n## Module Dependency Graph (Subset of Key Relationships)")
    print("```mermaid")
    print("graph TD")
    # Limit output for clarity
    count = 0
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if count < 50: # Only show first 50 for mermaid clarity
                print(f"    {node.replace('.', '_')} --> {neighbor.replace('.', '_')}")
                count += 1
    print("```")
    
    print("\n## Circular Dependencies Detected")
    if not cycles:
        print("✅ No circular dependencies detected.")
    else:
        for cycle in cycles:
            print(f"- ❌ {' -> '.join(cycle)}")
            
    print("\n## Orphaned Modules (No incoming references within documented core)")
    # Find nodes with no incoming edges
    all_imports = set()
    for neighbors in graph.values():
        for imp in neighbors:
            all_imports.add(imp)
            
    orphaned = [node for node in graph if node not in all_imports and node not in ['main', 'scripts.live']]
    for node in orphaned:
        # Check if it's a script (scripts are usually entry points)
        if not node.startswith('scripts.') and node != 'main' and not node.startswith('tests.'):
            print(f"- {node}")

    print("\n## External Dependencies Analysis")
    print("Check requirements.txt for versioning health.")
