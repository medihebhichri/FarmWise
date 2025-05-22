import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SoilService {
  private apiUrl = 'http://localhost:8080/api/soil';

  constructor(private http: HttpClient) {}

getSoil(lat: number, lon: number): Observable<any> {
  const params = { lat: lat.toString(), lon: lon.toString() };
  return this.http.get(this.apiUrl, { params });
}

}
