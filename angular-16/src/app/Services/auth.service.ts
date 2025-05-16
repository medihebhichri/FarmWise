import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Router } from '@angular/router';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private baseUrl = 'http://localhost:8080/api/v1/auth';

  constructor(private http: HttpClient, private router: Router) {}

  signup(userData: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/signup`, userData, { responseType: 'text' });
  }
  
  getUserRoles(): string[] {
    const roles = localStorage.getItem('roles');
    return roles ? JSON.parse(roles) : [];
  }
  
  hasRole(role: string): boolean {
    return this.getUserRoles().includes(role);
  }
  
  signin(credentials: any): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/signin`, credentials).pipe(
      tap((res) => {
        console.log('Login response:', res); // ✅ Add this
          localStorage.setItem('token', res.token);
        
        // Required:
        localStorage.setItem('username', res.username);
        localStorage.setItem('roles', JSON.stringify(res.roles)); // ✅ Must be here
      })
    );
  }
  getAllUsers(): Observable<any[]> {
    return this.http.get<any[]>('http://localhost:8080/api/v1/users');
  }
  
  
  getRoles(): string[] {
    const roles = localStorage.getItem('roles');
    return roles ? JSON.parse(roles) : [];
  }
  
  
  getUserDetails(): Observable<any> {
    return this.http.get(`${this.baseUrl}/user`);
  }

  saveToken(token: string, tokenType: string): void {
    localStorage.setItem('token', token);
    localStorage.setItem('tokenType', tokenType);
  }

  getToken(): string | null {
    return localStorage.getItem('token');
  }

  getTokenType(): string | null {
    return localStorage.getItem('tokenType');
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }

  logout(): void {
    localStorage.clear();
    this.router.navigate(['/login']);
  }
}

