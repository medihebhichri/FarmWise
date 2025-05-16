import { Component, OnInit } from '@angular/core';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { AuthService } from 'src/app/Services/auth.service';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.scss'],
  standalone: false,
})
export class ProfileComponent implements OnInit {
  user: any;
  safeImageUrl: SafeUrl | undefined;

  constructor(
    private authService: AuthService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {
    this.authService.getUserDetails().subscribe({
      next: (res) => {
        this.user = res;
        console.log('User:', this.user); // 🔍 Check if roles are present

        if (this.user?.imageUrl) {
          this.safeImageUrl = this.sanitizer.bypassSecurityTrustUrl(this.user.imageUrl);
        }
      },
      error: (err) => {
        console.error('Error fetching user details:', err);
      }
    });
  }

  getUserRoles(): string {
    if (!this.user?.roles || !Array.isArray(this.user.roles)) return 'No roles';
    return this.user.roles.map((r: any) => r.name).join(', ');
  }
}
