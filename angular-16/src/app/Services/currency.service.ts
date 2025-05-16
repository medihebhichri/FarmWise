// src/app/services/currency.service.ts
import { Injectable } from '@angular/core';
import { HttpClient }    from '@angular/common/http';
import { Observable, map } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class CurrencyService {
  // Call your own FastAPI proxy
private apiUrl = '/v6/latest/USD';

  constructor(private http: HttpClient) {}

  getUsdToTndRate(): Observable<number> {
    return this.http.get<{ rates: { [key: string]: number } }>(this.apiUrl).pipe(
      map(res => res.rates['TND']) // Extract the TND rate from the rates object
    );
  }
}
