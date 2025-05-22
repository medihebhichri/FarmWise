import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResponse {
  predicted_price: number;
}

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private apiUrl = 'http://localhost:4000/predict';

  constructor(private http: HttpClient) { }

  predict(features: number[]): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(this.apiUrl, { features });
  }
}

