import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CropService {
  private apiUrl = 'http://localhost:5000/predict';

  constructor(private http: HttpClient) {}

  predictCrop(input: any): Observable<any> {
    return this.http.post(this.apiUrl, input);
  }
}
