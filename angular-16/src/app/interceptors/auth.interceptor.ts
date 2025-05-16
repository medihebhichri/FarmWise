import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from '../Services/auth.service';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  constructor(private authService: AuthService) {}

  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = this.authService.getToken();

    // Exclude requests to the ClassificationService API
    if (req.url.includes('http://localhost:8000/predict/')) {
      console.log('Skipping AuthInterceptor for:', req.url);
      return next.handle(req);
    }

    if (token) {
      console.log('Token:', token); // Debugging log for token
      const cloned = req.clone({
        headers: req.headers.set('Authorization', `Bearer ${token}`)
      });
      console.log('Outgoing request:', cloned); // Debugging log for request
      return next.handle(cloned);
    }

    console.log('Outgoing request without token:', req); // Debugging log for requests without token
    return next.handle(req);
  }
  
}

