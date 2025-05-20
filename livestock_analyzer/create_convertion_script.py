#!/usr/bin/env python3
import os

def write_file(path, content):
    with open(path, 'w') as file:
        file.write(content)
    print(f"Created file: {path}")
    
def main():
    print("Creating automated conversion script...")
    
    # Create the main conversion script
    with open("convert_to_angular.py", "w") as f:
        f.write("""#!/usr/bin/env python3
import os
import shutil
import subprocess
import json
import glob
from pathlib import Path

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def write_file(path, content):
    with open(path, 'w') as file:
        file.write(content)
    print(f"Created file: {path}")

def run_command(command, cwd=None):
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        cwd=cwd
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error: {stderr.decode()}")
        return False
    return True

def create_angular_app():
    # Create main project directory
    project_path = "Angular_project"
    create_directory(project_path)
    
    # Create Angular frontend
    frontend_path = os.path.join(project_path, "livestock-frontend")
    if not os.path.exists(frontend_path):
        print("Setting up Angular frontend...")
        run_command(f"npm install -g @angular/cli")
        run_command(f"ng new livestock-frontend --routing=true --style=scss --skip-git", cwd=project_path)
        run_command("npm install @angular/material @angular/cdk chart.js @types/chart.js file-saver", cwd=frontend_path)
        run_command("ng add @angular/material --theme=indigo-pink --typography=true --animations=true", cwd=frontend_path)
    
    # Create FastAPI backend
    backend_path = os.path.join(project_path, "livestock-api")
    create_directory(backend_path)
    create_directory(os.path.join(backend_path, "app"))
    
    # Create requirements.txt for backend
    requirements_content = \"\"\"fastapi==0.95.0
uvicorn[standard]==0.21.1
pandas==2.0.0
numpy==1.24.2
python-multipart==0.0.6
scikit-learn==1.2.2
statsmodels==0.13.5
\"\"\"
    write_file(os.path.join(backend_path, "requirements.txt"), requirements_content)
    
    # Create data directory and copy data files
    create_directory(os.path.join(backend_path, "data"))
    for csv_file in glob.glob("data/*.csv"):
        shutil.copy2(csv_file, os.path.join(backend_path, "data"))
    
    return {
        "project_path": project_path,
        "frontend_path": frontend_path,
        "backend_path": backend_path
    }

def create_backend_files(backend_path):
    # Create main.py for FastAPI
    main_py_content = \"\"\"
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import shutil
import glob
from typing import List, Dict, Any
import json
from pathlib import Path

app = FastAPI(title="Livestock Analyzer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

@app.get("/")
def read_root():
    return {"message": "Livestock Analyzer API"}

@app.get("/datasets")
def get_datasets():
    files = glob.glob("data/*.csv")
    return [os.path.basename(file) for file in files]

@app.get("/dataset/{name}")
def get_dataset(name: str):
    try:
        file_path = f"data/{name}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Basic validation that it's a valid CSV
        try:
            pd.read_csv(file_path)
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        return {"filename": file.filename, "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/{name}/stats")
def get_dataset_stats(name: str):
    try:
        file_path = f"data/{name}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(file_path)
        
        # Calculate basic statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        stats = {}
        
        for col in numeric_columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": float(df[col].std())
            }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/{name}/correlation")
def get_correlation(name: str):
    try:
        file_path = f"data/{name}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr = numeric_df.corr().round(2)
        
        # Convert to format suitable for D3 heatmap
        columns = corr.columns.tolist()
        result = []
        
        for idx, col1 in enumerate(columns):
            for col2 in columns:
                result.append({
                    "group1": col1,
                    "group2": col2,
                    "value": corr.loc[col1, col2]
                })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
\"\"\"
    write_file(os.path.join(backend_path, "app", "main.py"), main_py_content)
    
    # Create start script for backend
    start_py_content = \"\"\"
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
\"\"\"
    write_file(os.path.join(backend_path, "start.py"), start_py_content)

def create_angular_components(frontend_path):
    # Create environment files
    environments_path = os.path.join(frontend_path, "src/environments")
    create_directory(environments_path)
    
    environment_ts = \"\"\"
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000'
};
\"\"\"
    write_file(os.path.join(environments_path, "environment.ts"), environment_ts)
    
    environment_prod_ts = \"\"\"
export const environment = {
  production: true,
  apiUrl: '/api'
};
\"\"\"
    write_file(os.path.join(environments_path, "environment.prod.ts"), environment_prod_ts)
    
    # Update angular.json to include environment paths
    angular_json_path = os.path.join(frontend_path, "angular.json")
    try:
        with open(angular_json_path, 'r') as f:
            angular_config = json.load(f)
        
        # Add environment file paths
        if 'projects' in angular_config and 'livestock-frontend' in angular_config['projects']:
            project = angular_config['projects']['livestock-frontend']
            if 'architect' in project and 'build' in project['architect']:
                options = project['architect']['build']['options']
                options['fileReplacements'] = [
                    {
                        "replace": "src/environments/environment.ts",
                        "with": "src/environments/environment.prod.ts"
                    }
                ]
                
                with open(angular_json_path, 'w') as f:
                    json.dump(angular_config, f, indent=2)
    except Exception as e:
        print(f"Error updating angular.json: {e}")
    
    # Generate components
    components = [
        "home", 
        "sidebar", 
        "data-explorer", 
        "analysis", 
        "forecasting", 
        "about",
        "upload-data"
    ]
    
    for component in components:
        run_command(f"ng generate component components/{component}", cwd=frontend_path)
    
    # Generate services
    services = ["data", "analysis", "forecasting"]
    for service in services:
        run_command(f"ng generate service services/{service}", cwd=frontend_path)
    
    # Create models
    models_path = os.path.join(frontend_path, "src/app/models")
    create_directory(models_path)
    
    # Create models/dataset.model.ts
    dataset_model = \"\"\"
export interface Dataset {
    name: string;
    path: string;
}
\"\"\"
    write_file(os.path.join(models_path, "dataset.model.ts"), dataset_model)

def update_angular_app_component(frontend_path):
    app_component_html = \"\"\"
<div class="app-container">
  <app-sidebar></app-sidebar>
  <main class="content">
    <router-outlet></router-outlet>
  </main>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/app.component.html"), app_component_html)
    
    app_component_scss = \"\"\"
.app-container {
  display: flex;
  height: 100vh;
  width: 100%;
}

.content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/app.component.scss"), app_component_scss)

def create_angular_routing(frontend_path):
    app_routing_module = \"\"\"
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { DataExplorerComponent } from './components/data-explorer/data-explorer.component';
import { AnalysisComponent } from './components/analysis/analysis.component';
import { ForecastingComponent } from './components/forecasting/forecasting.component';
import { AboutComponent } from './components/about/about.component';

const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'data-explorer', component: DataExplorerComponent },
  { path: 'analysis', component: AnalysisComponent },
  { path: 'forecasting', component: ForecastingComponent },
  { path: 'about', component: AboutComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/app-routing.module.ts"), app_routing_module)

def create_angular_services(frontend_path):
    # Data service
    data_service = \"\"\"
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiUrl = environment.apiUrl;
  
  constructor(private http: HttpClient) { }
  
  getDatasets(): Observable<string[]> {
    return this.http.get<string[]>(`${this.apiUrl}/datasets`);
  }
  
  getDataset(name: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/dataset/${name}`);
  }
  
  uploadDataset(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/upload-dataset`, formData);
  }
  
  getDatasetStats(name: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/dataset/${name}/stats`);
  }
  
  getCorrelationData(name: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/dataset/${name}/correlation`);
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/services/data.service.ts"), data_service)

    # Analysis service
    analysis_service = \"\"\"
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AnalysisService {
  private apiUrl = environment.apiUrl;
  
  constructor(private http: HttpClient) { }
  
  // Additional analysis methods will be added as needed
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/services/analysis.service.ts"), analysis_service)

    # Forecasting service
    forecasting_service = \"\"\"
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ForecastingService {
  private apiUrl = environment.apiUrl;
  
  constructor(private http: HttpClient) { }
  
  // Additional forecasting methods will be added as needed
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/services/forecasting.service.ts"), forecasting_service)

def create_sidebar_component(frontend_path):
    # HTML Template
    sidebar_html = \"\"\"
<div class="sidebar-container">
  <h1 class="app-title">Livestock Analyzer</h1>
  
  <div class="navigation">
    <h2>Navigation</h2>
    <mat-nav-list>
      <a mat-list-item routerLink="/" routerLinkActive="active-link" [routerLinkActiveOptions]="{exact: true}">
        <mat-icon>home</mat-icon> Home
      </a>
      <a mat-list-item routerLink="/data-explorer" routerLinkActive="active-link">
        <mat-icon>table_chart</mat-icon> Data Explorer
      </a>
      <a mat-list-item routerLink="/analysis" routerLinkActive="active-link">
        <mat-icon>insights</mat-icon> Analysis
      </a>
      <a mat-list-item routerLink="/forecasting" routerLinkActive="active-link">
        <mat-icon>trending_up</mat-icon> Forecasting
      </a>
      <a mat-list-item routerLink="/about" routerLinkActive="active-link">
        <mat-icon>info</mat-icon> About
      </a>
    </mat-nav-list>
  </div>
  
  <div class="dataset-section" *ngIf="showDatasetSelection()">
    <h2>Data Selection</h2>
    
    <mat-form-field appearance="fill" *ngIf="datasets.length > 0">
      <mat-label>Select Dataset</mat-label>
      <mat-select [(value)]="selectedDataset" (selectionChange)="onDatasetChange()">
        <mat-option *ngFor="let dataset of datasets" [value]="dataset">
          {{dataset}}
        </mat-option>
      </mat-select>
    </mat-form-field>
    
    <div *ngIf="datasets.length === 0" class="warning-message">
      No datasets found. Please upload a dataset.
    </div>
    
    <button mat-raised-button color="primary" (click)="openUploadDialog()">
      <mat-icon>upload</mat-icon> Upload Dataset
    </button>
  </div>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/sidebar/sidebar.component.html"), sidebar_html)

    # Component Styles
    sidebar_scss = \"\"\"
.sidebar-container {
  width: 250px;
  height: 100%;
  background-color: #f5f5f5;
  padding: 16px;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
}

.app-title {
  margin-bottom: 24px;
  color: #3f51b5;
}

.navigation {
  margin-bottom: 24px;
}

.dataset-section {
  margin-top: auto;
}

mat-form-field {
  width: 100%;
  margin-bottom: 16px;
}

.active-link {
  background-color: rgba(63, 81, 181, 0.1);
  color: #3f51b5;
}

.warning-message {
  color: #f44336;
  margin-bottom: 16px;
  font-size: 14px;
}

button {
  width: 100%;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/sidebar/sidebar.component.scss"), sidebar_scss)

    # Component TypeScript
    sidebar_ts = \"\"\"
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MatDialog } from '@angular/material/dialog';
import { DataService } from '../../services/data.service';
import { UploadDataComponent } from '../upload-data/upload-data.component';

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent implements OnInit {
  datasets: string[] = [];
  selectedDataset: string | null = null;
  
  constructor(
    private dataService: DataService,
    private router: Router,
    private dialog: MatDialog
  ) { }
  
  ngOnInit(): void {
    this.loadDatasets();
    
    // Restore selected dataset from localStorage if available
    const savedDataset = localStorage.getItem('selectedDataset');
    if (savedDataset) {
      this.selectedDataset = savedDataset;
    }
  }
  
  loadDatasets(): void {
    this.dataService.getDatasets().subscribe({
      next: (datasets) => {
        this.datasets = datasets;
        
        // If there's a selected dataset but it's not in the list, reset it
        if (this.selectedDataset && !this.datasets.includes(this.selectedDataset)) {
          this.selectedDataset = null;
          localStorage.removeItem('selectedDataset');
        }
      },
      error: (error) => {
        console.error('Error loading datasets', error);
      }
    });
  }
  
  onDatasetChange(): void {
    if (this.selectedDataset) {
      localStorage.setItem('selectedDataset', this.selectedDataset);
      
      // Refresh current route to update the view with the new dataset
      const currentUrl = this.router.url;
      this.router.navigateByUrl('/', { skipLocationChange: true }).then(() => {
        this.router.navigate([currentUrl]);
      });
    }
  }
  
  showDatasetSelection(): boolean {
    // Only show dataset selection on Data Explorer, Analysis, and Forecasting pages
    const url = this.router.url;
    return url.includes('/data-explorer') || url.includes('/analysis') || url.includes('/forecasting');
  }
  
  openUploadDialog(): void {
    const dialogRef = this.dialog.open(UploadDataComponent, {
      width: '500px'
    });
    
    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        // Reload datasets if a new one was uploaded
        this.loadDatasets();
        
        // If a new dataset was uploaded and returned, select it
        if (result.dataset) {
          this.selectedDataset = result.dataset;
          this.onDatasetChange();
        }
      }
    });
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/sidebar/sidebar.component.ts"), sidebar_ts)

def create_upload_component(frontend_path):
    upload_html = \"\"\"
<h2 mat-dialog-title>Upload Dataset</h2>

<mat-dialog-content>
  <div class="upload-container">
    <p>Please select a CSV file to upload:</p>
    
    <div class="file-input">
      <button mat-raised-button color="primary" (click)="fileInput.click()">
        <mat-icon>attach_file</mat-icon> Select File
      </button>
      <input hidden type="file" #fileInput (change)="onFileSelected($event)" accept=".csv">
      <span class="filename" *ngIf="selectedFile">{{ selectedFile.name }}</span>
    </div>
    
    <mat-progress-bar *ngIf="uploading" mode="indeterminate"></mat-progress-bar>
    
    <div *ngIf="uploadError" class="error-message">
      {{ uploadError }}
    </div>
  </div>
</mat-dialog-content>

<mat-dialog-actions align="end">
  <button mat-button [mat-dialog-close]="null">Cancel</button>
  <button mat-raised-button color="primary" 
          [disabled]="!selectedFile || uploading"
          (click)="uploadFile()">Upload</button>
</mat-dialog-actions>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/upload-data/upload-data.component.html"), upload_html)

    upload_scss = \"\"\"
.upload-container {
  padding: 16px 0;
}

.file-input {
  display: flex;
  align-items: center;
  margin: 16px 0;
}

.filename {
  margin-left: 16px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

mat-progress-bar {
  margin: 16px 0;
}

.error-message {
  color: #f44336;
  margin-top: 16px;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/upload-data/upload-data.component.scss"), upload_scss)

    upload_ts = \"\"\"
import { Component } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';
import { DataService } from '../../services/data.service';

@Component({
  selector: 'app-upload-data',
  templateUrl: './upload-data.component.html',
  styleUrls: ['./upload-data.component.scss']
})
export class UploadDataComponent {
  selectedFile: File | null = null;
  uploading = false;
  uploadError: string | null = null;

  constructor(
    private dialogRef: MatDialogRef<UploadDataComponent>,
    private dataService: DataService
  ) { }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.uploadError = null;
    }
  }

  uploadFile(): void {
    if (!this.selectedFile) {
      return;
    }

    this.uploading = true;
    this.uploadError = null;

    this.dataService.uploadDataset(this.selectedFile).subscribe({
      next: (response) => {
        this.uploading = false;
        this.dialogRef.close({
          success: true,
          dataset: this.selectedFile?.name
        });
      },
      error: (error) => {
        this.uploading = false;
        this.uploadError = error.error?.detail || 'An error occurred during upload';
        console.error('Upload error', error);
      }
    });
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/upload-data/upload-data.component.ts"), upload_ts)

def create_home_component(frontend_path):
    home_html = \"\"\"
<div class="home-container">
  <header class="header">
    <h1>Welcome to Livestock Analyzer</h1>
    <p class="subtitle">
      Your comprehensive tool for analyzing and forecasting livestock data
    </p>
  </header>
  
  <div class="features">
    <mat-card class="feature-card">
      <mat-card-header>
        <mat-icon mat-card-avatar>search</mat-icon>
        <mat-card-title>Data Explorer</mat-card-title>
      </mat-card-header>
      <mat-card-content>
        <p>
          Explore your livestock datasets with interactive tables and visualizations.
          Filter, sort, and analyze your data with ease.
        </p>
      </mat-card-content>
      <mat-card-actions>
        <button mat-button color="primary" routerLink="/data-explorer">EXPLORE DATA</button>
      </mat-card-actions>
    </mat-card>
    
    <mat-card class="feature-card">
      <mat-card-header>
        <mat-icon mat-card-avatar>insights</mat-icon>
        <mat-card-title>Data Analysis</mat-card-title>
      </mat-card-header>
      <mat-card-content>
        <p>
          Generate statistical analyses, correlation matrices, and key metrics for your livestock data.
          Discover trends and patterns in your datasets.
        </p>
      </mat-card-content>
      <mat-card-actions>
        <button mat-button color="primary" routerLink="/analysis">ANALYZE DATA</button>
      </mat-card-actions>
    </mat-card>
    
    <mat-card class="feature-card">
      <mat-card-header>
        <mat-icon mat-card-avatar>trending_up</mat-icon>
        <mat-card-title>Forecasting</mat-card-title>
      </mat-card-header>
      <mat-card-content>
        <p>
          Predict future trends in your livestock data using advanced forecasting models.
          Make informed decisions based on data-driven predictions.
        </p>
      </mat-card-content>
      <mat-card-actions>
        <button mat-button color="primary" routerLink="/forecasting">FORECAST DATA</button>
      </mat-card-actions>
    </mat-card>
  </div>
  
  <div class="get-started">
    <h2>Get Started</h2>
    <p>
      To begin analyzing your livestock data, upload a dataset using the sidebar or explore our sample datasets.
    </p>
    <button mat-raised-button color="primary" (click)="openUploadDialog()">
      <mat-icon>upload</mat-icon> Upload Dataset
    </button>
  </div>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/home/home.component.html"), home_html)

    home_scss = \"\"\"
.home-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 40px;
}

.subtitle {
  font-size: 1.2em;
  color: #666;
}

.features {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 40px;
}

.feature-card {
  flex: 1;
  min-width: 300px;
}

.get-started {
  text-align: center;
  margin-top: 40px;
}

@media (max-width: 960px) {
  .features {
    flex-direction: column;
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/home/home.component.scss"), home_scss)

    home_ts = \"\"\"
import { Component } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { UploadDataComponent } from '../upload-data/upload-data.component';
import { Router } from '@angular/router';
import { DataService } from '../../services/data.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  constructor(
    private dialog: MatDialog,
    private router: Router,
    private dataService: DataService
  ) { }

  openUploadDialog(): void {
    const dialogRef = this.dialog.open(UploadDataComponent, {
      width: '500px'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.success && result.dataset) {
        localStorage.setItem('selectedDataset', result.dataset);
        this.router.navigate(['/data-explorer']);
      }
    });
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/home/home.component.ts"), home_ts)

def update_app_module(frontend_path):
    app_module = \"\"\"
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

// Angular Material Imports
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatSortModule } from '@angular/material/sort';
import { MatInputModule } from '@angular/material/input';
import { MatDialogModule } from '@angular/material/dialog';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

// Components
import { HomeComponent } from './components/home/home.component';
import { SidebarComponent } from './components/sidebar/sidebar.component';
import { DataExplorerComponent } from './components/data-explorer/data-explorer.component';
import { AnalysisComponent } from './components/analysis/analysis.component';
import { ForecastingComponent } from './components/forecasting/forecasting.component';
import { AboutComponent } from './components/about/about.component';
import { UploadDataComponent } from './components/upload-data/upload-data.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    SidebarComponent,
    DataExplorerComponent,
    AnalysisComponent,
    ForecastingComponent,
    AboutComponent,
    UploadDataComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    BrowserAnimationsModule,
    FormsModule,
    ReactiveFormsModule,
    MatSidenavModule,
    MatToolbarModule,
    MatListModule,
    MatIconModule,
    MatButtonModule,
    MatCardModule,
    MatFormFieldModule,
    MatSelectModule,
    MatProgressBarModule,
    MatTableModule,
    MatPaginatorModule,
    MatSortModule,
    MatInputModule,
    MatDialogModule,
    MatTabsModule,
    MatProgressSpinnerModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/app.module.ts"), app_module)

def create_additional_components(frontend_path):
    # Data Explorer Component
    data_explorer_html = \"\"\"
<div class="data-explorer-container">
  <h1>Data Explorer</h1>

  <div *ngIf="!selectedDataset" class="no-data-message">
    <p>Please select a dataset from the sidebar to begin exploring.</p>
    <button mat-raised-button color="primary" (click)="openUploadDialog()">
      <mat-icon>upload</mat-icon> Upload Dataset
    </button>
  </div>

  <div *ngIf="selectedDataset && !dataLoaded" class="loading">
    <mat-spinner diameter="40"></mat-spinner>
    <p>Loading dataset...</p>
  </div>

  <div *ngIf="selectedDataset && dataLoaded" class="data-content">
    <h2>{{ selectedDataset }}</h2>

    <mat-form-field appearance="outline">
      <mat-label>Filter</mat-label>
      <input matInput (keyup)="applyFilter($event)" placeholder="Search data" #input>
      <mat-icon matSuffix>search</mat-icon>
    </mat-form-field>

    <div class="table-container">
      <table mat-table [dataSource]="dataSource" matSort>
        <ng-container *ngFor="let column of displayedColumns" [matColumnDef]="column">
          <th mat-header-cell *matHeaderCellDef mat-sort-header>{{ column }}</th>
          <td mat-cell *matCellDef="let element">{{ element[column] }}</td>
        </ng-container>

        <tr mat-header-row *matHeaderRowDef="displayedColumns; sticky: true"></tr>
        <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>

        <tr class="mat-row" *matNoDataRow>
          <td class="mat-cell" colspan="4">No data matching the filter "{{input.value}}"</td>
        </tr>
      </table>

      <mat-paginator [pageSize]="10" [pageSizeOptions]="[5, 10, 25, 50]" showFirstLastButtons></mat-paginator>
    </div>
  </div>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/data-explorer/data-explorer.component.html"), data_explorer_html)

    data_explorer_scss = \"\"\"
.data-explorer-container {
  padding: 20px;
}

.no-data-message {
  text-align: center;
  padding: 40px 0;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
}

.data-content {
  margin-top: 20px;
}

mat-form-field {
  width: 100%;
  margin-bottom: 20px;
}

.table-container {
  position: relative;
  max-height: 600px;
  overflow: auto;
}

table {
  width: 100%;
}

tr.mat-row:hover {
  background-color: #f5f5f5;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/data-explorer/data-explorer.component.scss"), data_explorer_scss)

    data_explorer_ts = \"\"\"
import { Component, OnInit, ViewChild } from '@angular/core';
import { MatTableDataSource } from '@angular/material/table';
import { MatPaginator } from '@angular/material/paginator';
import { MatSort } from '@angular/material/sort';
import { MatDialog } from '@angular/material/dialog';
import { DataService } from '../../services/data.service';
import { UploadDataComponent } from '../upload-data/upload-data.component';

@Component({
  selector: 'app-data-explorer',
  templateUrl: './data-explorer.component.html',
  styleUrls: ['./data-explorer.component.scss']
})
export class DataExplorerComponent implements OnInit {
  selectedDataset: string | null = null;
  dataSource = new MatTableDataSource<any>([]);
  displayedColumns: string[] = [];
  dataLoaded = false;

  @ViewChild(MatPaginator) paginator!: MatPaginator;
  @ViewChild(MatSort) sort!: MatSort;

  constructor(
    private dataService: DataService,
    private dialog: MatDialog
  ) { }

  ngOnInit(): void {
    // Get selected dataset from localStorage
    this.selectedDataset = localStorage.getItem('selectedDataset');
    if (this.selectedDataset) {
      this.loadDataset();
    }
  }

  loadDataset(): void {
    if (!this.selectedDataset) return;

    this.dataLoaded = false;
    
    this.dataService.getDataset(this.selectedDataset).subscribe({
      next: (data) => {
        if (data && data.length > 0) {
          // Get column names from the first data row
          this.displayedColumns = Object.keys(data[0]);
          
          // Set data to table
          this.dataSource.data = data;
          
          // Set up sorting and pagination after data loads
          setTimeout(() => {
            this.dataSource.paginator = this.paginator;
            this.dataSource.sort = this.sort;
          });
        }
        this.dataLoaded = true;
      },
      error: (error) => {
        console.error('Error loading dataset', error);
        this.dataLoaded = true;
      }
    });
  }

  applyFilter(event: Event): void {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();

    if (this.dataSource.paginator) {
      this.dataSource.paginator.firstPage();
    }
  }

  openUploadDialog(): void {
    const dialogRef = this.dialog.open(UploadDataComponent, {
      width: '500px'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.success && result.dataset) {
        this.selectedDataset = result.dataset;
        localStorage.setItem('selectedDataset', this.selectedDataset);
        this.loadDataset();
      }
    });
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/data-explorer/data-explorer.component.ts"), data_explorer_ts)

    # Create about component
    about_html = \"\"\"
<div class="about-container">
  <h1>About Livestock Analyzer</h1>
  
  <mat-card>
    <mat-card-content>
      <p>
        Livestock Analyzer is a comprehensive tool designed to help livestock farmers,
        researchers, and agricultural professionals analyze and forecast livestock data.
      </p>
      
      <p>
        This application was built with Angular for the frontend and FastAPI for the backend,
        providing a modern, responsive, and powerful platform for livestock data analysis.
      </p>
      
      <h3>Features</h3>
      <ul>
        <li>Import and explore livestock datasets</li>
        <li>Analyze and visualize livestock data</li>
        <li>Generate statistical insights</li>
        <li>Create forecasts based on historical data</li>
        <li>Export results and findings</li>
      </ul>
      
      <h3>Contact</h3>
      <p>
        For support, feature requests, or more information, please contact:
        <a href="mailto:support@livestock-analyzer.com">support@livestock-analyzer.com</a>
      </p>
    </mat-card-content>
  </mat-card>
  
  <h2>Version Information</h2>
  <div class="version-info">
    <p><strong>Version:</strong> 1.0.0</p>
    <p><strong>Last Updated:</strong> {{ currentDate | date:'longDate' }}</p>
  </div>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/about/about.component.html"), about_html)

    about_scss = \"\"\"
.about-container {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

mat-card {
  margin: 20px 0;
}

h1 {
  margin-bottom: 20px;
}

h2 {
  margin-top: 40px;
}

.version-info {
  background-color: #f5f5f5;
  padding: 16px;
  border-radius: 4px;
}

ul {
  padding-left: 20px;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/about/about.component.scss"), about_scss)

    about_ts = \"\"\"
import { Component } from '@angular/core';

@Component({
  selector: 'app-about',
  templateUrl: './about.component.html',
  styleUrls: ['./about.component.scss']
})
export class AboutComponent {
  currentDate = new Date();
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/about/about.component.ts"), about_ts)

def create_analysis_component(frontend_path):
    analysis_html = \"\"\"
<div class="analysis-container">
  <h1>Data Analysis</h1>

  <div *ngIf="!selectedDataset" class="no-data-message">
    <p>Please select a dataset from the sidebar to begin analysis.</p>
    <button mat-raised-button color="primary" (click)="openUploadDialog()">
      <mat-icon>upload</mat-icon> Upload Dataset
    </button>
  </div>

  <div *ngIf="selectedDataset && loading" class="loading">
    <mat-spinner diameter="40"></mat-spinner>
    <p>Loading analysis data...</p>
  </div>

  <div *ngIf="selectedDataset && !loading" class="analysis-content">
    <h2>Analysis for: {{ selectedDataset }}</h2>

    <mat-tab-group>
      <mat-tab label="Summary Statistics">
        <div class="stats-container" *ngIf="statistics">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th *ngFor="let column of statisticsColumns">{{ column }}</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let metric of ['mean', 'median', 'min', 'max', 'std']">
                <td>{{ metric }}</td>
                <td *ngFor="let column of statisticsColumns">
                  {{ statistics[column][metric] | number:'1.2-2' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div *ngIf="!statistics" class="no-data-message">
          No statistics available for this dataset.
        </div>
      </mat-tab>
      
      <mat-tab label="Correlation Matrix">
        <div class="correlation-container">
          <p>Correlation matrix visualization will be implemented here.</p>
        </div>
      </mat-tab>
    </mat-tab-group>
  </div>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/analysis/analysis.component.html"), analysis_html)

    analysis_scss = \"\"\"
.analysis-container {
  padding: 20px;
}

.no-data-message {
  text-align: center;
  padding: 40px 0;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
}

.analysis-content {
  margin-top: 20px;
}

mat-tab-group {
  margin-top: 20px;
}

.stats-container {
  overflow-x: auto;
  margin: 20px 0;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 10px;
  border: 1px solid #ddd;
  text-align: left;
}

th {
  background-color: #f5f5f5;
  font-weight: 500;
}

tr:nth-child(even) {
  background-color: #f9f9f9;
}

.correlation-container {
  margin: 20px 0;
  min-height: 300px;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/analysis/analysis.component.scss"), analysis_scss)

    analysis_ts = \"\"\"
import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { DataService } from '../../services/data.service';
import { UploadDataComponent } from '../upload-data/upload-data.component';

@Component({
  selector: 'app-analysis',
  templateUrl: './analysis.component.html',
  styleUrls: ['./analysis.component.scss']
})
export class AnalysisComponent implements OnInit {
  selectedDataset: string | null = null;
  loading = false;
  statistics: any = null;
  statisticsColumns: string[] = [];
  correlationData: any[] = [];

  constructor(
    private dataService: DataService,
    private dialog: MatDialog
  ) { }

  ngOnInit(): void {
    // Get selected dataset from localStorage
    this.selectedDataset = localStorage.getItem('selectedDataset');
    if (this.selectedDataset) {
      this.loadAnalysisData();
    }
  }

  loadAnalysisData(): void {
    if (!this.selectedDataset) return;

    this.loading = true;
    
    // Load statistics
    this.dataService.getDatasetStats(this.selectedDataset).subscribe({
      next: (stats) => {
        this.statistics = stats;
        this.statisticsColumns = Object.keys(stats);
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading statistics', error);
        this.loading = false;
      }
    });
    
    // Load correlation data
    this.dataService.getCorrelationData(this.selectedDataset).subscribe({
      next: (data) => {
        this.correlationData = data;
        // Correlation visualization would be implemented here
      },
      error: (error) => {
        console.error('Error loading correlation data', error);
      }
    });
  }

  openUploadDialog(): void {
    const dialogRef = this.dialog.open(UploadDataComponent, {
      width: '500px'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.success && result.dataset) {
        this.selectedDataset = result.dataset;
        localStorage.setItem('selectedDataset', this.selectedDataset);
        this.loadAnalysisData();
      }
    });
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/analysis/analysis.component.ts"), analysis_ts)

def create_forecasting_component(frontend_path):
    forecasting_html = \"\"\"
<div class="forecasting-container">
  <h1>Forecasting</h1>

  <div *ngIf="!selectedDataset" class="no-data-message">
    <p>Please select a dataset from the sidebar to begin forecasting.</p>
    <button mat-raised-button color="primary" (click)="openUploadDialog()">
      <mat-icon>upload</mat-icon> Upload Dataset
    </button>
  </div>

  <div *ngIf="selectedDataset" class="forecasting-content">
    <h2>Forecasting for: {{ selectedDataset }}</h2>
    
    <p class="feature-notice">
      <mat-icon>info</mat-icon>
      Forecasting functionality will be implemented in the next phase of development.
    </p>
    
    <mat-card>
      <mat-card-header>
        <mat-card-title>Forecasting Options</mat-card-title>
      </mat-card-header>
      <mat-card-content>
        <p>
          This section will provide forecasting capabilities using various time series models
          including ARIMA, Exponential Smoothing, and Prophet to predict future trends in your
          livestock data.
        </p>
      </mat-card-content>
    </mat-card>
  </div>
</div>
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/forecasting/forecasting.component.html"), forecasting_html)

    forecasting_scss = \"\"\"
.forecasting-container {
  padding: 20px;
}

.no-data-message {
  text-align: center;
  padding: 40px 0;
}

.forecasting-content {
  margin-top: 20px;
}

.feature-notice {
  display: flex;
  align-items: center;
  background-color: #fff3e0;
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.feature-notice mat-icon {
  color: #ff9800;
  margin-right: 10px;
}

mat-card {
  margin-bottom: 20px;
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/forecasting/forecasting.component.scss"), forecasting_scss)

    forecasting_ts = \"\"\"
import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { UploadDataComponent } from '../upload-data/upload-data.component';

@Component({
  selector: 'app-forecasting',
  templateUrl: './forecasting.component.html',
  styleUrls: ['./forecasting.component.scss']
})
export class ForecastingComponent implements OnInit {
  selectedDataset: string | null = null;

  constructor(private dialog: MatDialog) { }

  ngOnInit(): void {
    // Get selected dataset from localStorage
    this.selectedDataset = localStorage.getItem('selectedDataset');
  }

  openUploadDialog(): void {
    const dialogRef = this.dialog.open(UploadDataComponent, {
      width: '500px'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.success && result.dataset) {
        this.selectedDataset = result.dataset;
        localStorage.setItem('selectedDataset', this.selectedDataset);
      }
    });
  }
}
\"\"\"
    write_file(os.path.join(frontend_path, "src/app/components/forecasting/forecasting.component.ts"), forecasting_ts)

def create_readme(project_path):
    readme_content = \"\"\"
# Livestock Analyzer - Angular Version

This project is an Angular implementation of the Livestock Analyzer application, converted from a Streamlit application.

## Project Structure

The project consists of two main parts:

1. **Frontend**: An Angular application with Material UI
2. **Backend**: A FastAPI-based REST API

## Getting Started

### Running the Backend

1. Navigate to the livestock-api directory:
```bash
cd livestock-api
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python start.py
```

The backend API will be available at http://localhost:8000

### Running the Frontend

1. Navigate to the livestock-frontend directory:
```bash
cd livestock-frontend
```

2. Install the required dependencies:
```bash
npm install
```

3. Start the Angular development server:
```bash
ng serve
```

The application will be available at http://localhost:4200

## Features

- Data exploration with sorting, filtering, and pagination
- Dataset uploading
- Basic statistical analysis
- Correlation analysis
- (Coming Soon) Forecasting capabilities

## Development

### Adding New Features

- Backend: Add new endpoints to `livestock-api/app/main.py`
- Frontend: Extend the Angular components and services as needed

## Building for Production

### Frontend

```bash
cd livestock-frontend
ng build --prod
```

### Backend

The FastAPI backend can be deployed using various methods such as:
- Gunicorn with Uvicorn workers
- Docker containers
- Cloud services like AWS Lambda or Google Cloud Functions

## License

MIT
\"\"\"
    write_file(os.path.join(project_path, "README.md"), readme_content)
""")
    
    # Create run_conversion.py - the script that will execute the conversion
    run_conversion_content = """#!/usr/bin/env python3
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
    print(f"\\nYour Angular project has been created in the '{project_path}' directory.")
    print("\\nTo start the backend:")
    print(f"  cd {backend_path}")
    print("  pip install -r requirements.txt")
    print("  python start.py")
    print("\\nTo start the frontend:")
    print(f"  cd {frontend_path}")
    print("  npm install")
    print("  ng serve")
    print("\\nThen open your browser to http://localhost:4200")
    
if __name__ == "__main__":
    main()
"""
    write_file("run_conversion.py", run_conversion_content)
    os.chmod("run_conversion.py", 0o755)  # Make executable
    
    print("Automation scripts created successfully!")
    print("\nTo convert your Streamlit app to Angular, run:")
    print("  python run_conversion.py")

if __name__ == "__main__":
    main()
