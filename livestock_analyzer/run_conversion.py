#!/usr/bin/env python3
import os
import sys
import subprocess

# Import the conversion functions
from convert_to_angular import (
    create_angular_app,
    create_backend_files,
    create_angular_components,
    update_angular_app_component,
    create_angular_routing,
    create_angular_services,
    create_sidebar_component,
    create_upload_component,
    create_home_component,
    update_app_module,
    create_additional_components,
    create_analysis_component,
    create_forecasting_component,
    create_readme
)

def main():
    print("Starting Streamlit to Angular conversion...")
    
    # Create Angular project structure
    paths = create_angular_app()
    project_path = paths['project_path']
    frontend_path = paths['frontend_path']
    backend_path = paths['backend_path']
    
    print("Creating backend API files...")
    create_backend_files(backend_path)
    
    print("Creating Angular components...")
    create_angular_components(frontend_path)
    
    print("Updating Angular app component...")
    update_angular_app_component(frontend_path)
    
    print("Creating Angular routing...")
    create_angular_routing(frontend_path)
    
    print("Creating Angular services...")
    create_angular_services(frontend_path)
    
    print("Creating sidebar component...")
    create_sidebar_component(frontend_path)
    
    print("Creating upload component...")
    create_upload_component(frontend_path)
    
    print("Creating home component...")
    create_home_component(frontend_path)
    
    print("Updating app module...")
    update_app_module(frontend_path)
    
    print("Creating additional components...")
    create_additional_components(frontend_path)
    
    print("Creating analysis component...")
    create_analysis_component(frontend_path)
    
    print("Creating forecasting component...")
    create_forecasting_component(frontend_path)
    
    print("Creating project README...")
    create_readme(project_path)
    
    print("Conversion complete!")
    print(f"\nYour Angular project has been created in the '{project_path}' directory.")
    print("\nTo start the backend:")
    print(f"  cd {backend_path}")
    print("  pip install -r requirements.txt")
    print("  python start.py")
    print("\nTo start the frontend:")
    print(f"  cd {frontend_path}")
    print("  npm install")
    print("  ng serve")
    print("\nThen open your browser to http://localhost:4200")
    
if __name__ == "__main__":
    main()
