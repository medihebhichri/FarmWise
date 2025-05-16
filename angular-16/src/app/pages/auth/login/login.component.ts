import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/Services/auth.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  standalone: false,
  styleUrls: ['./login.component.scss'], 
})
export class LoginComponent {
  username = '';
  password = '';
  errorMessage = '';
  form: FormGroup;

  constructor(private authService: AuthService, private router: Router, private fb: FormBuilder) {
    this.form = this.fb.group({
      email: ['', [Validators.required]],
      password: ['', [Validators.required]]
    });
  }

  onLogin(): void {
    const credentials = { username: this.username, password: this.password };
  
    this.authService.signin(credentials).subscribe({
      next: (res) => {
        localStorage.setItem('token', res.token);
        localStorage.setItem('username', res.username);
        localStorage.setItem('roles', JSON.stringify(res.roles));
  
        const roleNames: string[] = res.roles;
  
        if (roleNames.includes('ROLE_ADMIN')) {
          this.router.navigate(['/dashboard']);
        } else if (roleNames.includes('ROLE_USER')) {
          this.router.navigate(['/dashboard']);
        } else {
          this.router.navigate(['/']);
        }
      },
      error: () => {
        this.errorMessage = 'Invalid username or password';
      }
    });
  }

  onSubmit(): void {
    if (this.form.valid) {
      const { email, password } = this.form.value;
      this.username = email;
      this.password = password;
      this.onLogin();
    } else {
      this.errorMessage = 'Please fill in all required fields correctly.';
    }
  }
}
