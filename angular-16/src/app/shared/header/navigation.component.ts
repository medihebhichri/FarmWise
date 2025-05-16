import { Component, AfterViewInit, OnInit, EventEmitter, Output } from '@angular/core';
import { NgbDropdownModule, NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { AuthService } from 'src/app/Services/auth.service';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-navigation',
  standalone: true,
  imports: [NgbDropdownModule],
  templateUrl: './navigation.component.html'
})
export class NavigationComponent implements AfterViewInit, OnInit {
  @Output() toggleSidebar = new EventEmitter<void>();

  public showSearch = false;
  user: any;
  safeImageUrl: SafeUrl | string = 'assets/images/users/default.jpg'; // fallback image

  constructor(
    private modalService: NgbModal,
    private authService: AuthService,
    private sanitizer: DomSanitizer
  ) {}
  logout(): void {
    this.authService.logout();
    window.location.href = 'auth/login'; // Redirect to the login page
  }
  ngOnInit(): void {
    this.authService.getUserDetails().subscribe({
      next: (res) => {
        this.user = res;
        if (this.user?.imageUrl) {
          this.safeImageUrl = this.sanitizer.bypassSecurityTrustUrl(this.user.imageUrl);
        }
      },
      error: (err) => console.error('Failed to fetch user info:', err)
    });
  }

  ngAfterViewInit() {}
}
