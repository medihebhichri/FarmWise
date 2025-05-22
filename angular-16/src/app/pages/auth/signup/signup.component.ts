import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/Services/auth.service';

@Component({
  selector: 'app-signup',
  templateUrl: './signup.component.html',
  standalone: false,
  styleUrls: ['./signup.component.scss'],
})
export class SignupComponent {
  userData: any = {
    firstName: '',
    lastName: '',
    username: '',
    password: '',
    imageUrl: ''   // 🔥 Add this field
  };

  constructor(private authService: AuthService, private router: Router) {}

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.userData.imageUrl = e.target.result; // FULL Data URL
      };
      reader.readAsDataURL(file); // ⚡️ NOT readAsText!
    }
  }
  

  onSignup(): void {
    console.log('Payload being sent:', this.userData);
    this.authService.signup(this.userData).subscribe({
      next: () => this.router.navigate(['/login']),
      error: (err) => console.error('Signup error:', err)
    });
  }
}
