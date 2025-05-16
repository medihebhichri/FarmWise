// src/app/services/classification.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResponse {
  prediction: string;
}

@Injectable({ providedIn: 'root' })
export class ClassificationService {
  // Adjust this to your FastAPI host/port in production
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

 predict(lat: number, lon: number): Observable<{prediction: string, tile: string}> {
  const params = new HttpParams()
    .set('lat', lat.toString())
    .set('lon', lon.toString());
  return this.http.get<{prediction: string, tile: string}>(`${this.apiUrl}/predict/`, { params });
}

}
