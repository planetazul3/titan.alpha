import importlib
import os
import sys

# Add the current directory to sys.path to allow internal imports
sys.path.append(os.getcwd())

failed_imports = []
successful_imports = []
warnings: list[str] = []

# Directory to scan
target_dir = 'x.titan'
# If we are already in x.titan, just scan the current directory but ignore venv
root_search = '.'

for root, dirs, files in os.walk(root_search):
    if 'venv' in root or '.git' in root or '__pycache__' in root or 'python-deriv-api' in root:
        continue
        
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            module_path = os.path.join(root, file)
            # Convert path to module name
            module_name = module_path.replace('./', '').replace('/', '.').replace('.py', '')
            
            try:
                importlib.import_module(module_name)
                successful_imports.append(module_name)
            except Exception as e:
                failed_imports.append((module_name, str(e)))

# Generate detailed report
with open('/home/planetazul3/.gemini/antigravity/brain/46bc7d97-9458-4807-8102-478ba90e901f/IMPORT_VALIDATION.md', 'w') as f:
    f.write("# IMPORT_VALIDATION.md\n\n")
    f.write(f"## Statistics\n")
    total = len(successful_imports) + len(failed_imports)
    success_rate = (len(successful_imports) / total * 100) if total > 0 else 0
    f.write(f"- ✅ Successful imports: {len(successful_imports)}\n")
    f.write(f"- ❌ Failed imports: {len(failed_imports)}\n")
    f.write(f"- 📊 Import success rate: {success_rate:.2f}%\n\n")
    
    if failed_imports:
        f.write("## ❌ Failed Imports\n")
        f.write("| Module | Error |\n")
        f.write("|--------|-------|\n")
        for mod, err in failed_imports:
            f.write(f"| {mod} | {err} |\n")
        f.write("\n")
    else:
        f.write("## ✅ All modules imported successfully!\n")

print(f"Import validation complete. Success rate: {success_rate:.2f}%")
