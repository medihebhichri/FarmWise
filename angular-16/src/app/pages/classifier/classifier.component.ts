// src/app/components/classifier/classifier.component.ts
import { Component, AfterViewInit } from '@angular/core';
import * as L from 'leaflet';
import { ClassificationService } from 'src/app/Services/classification.service';

@Component({
  selector: 'app-classifier',
  templateUrl: './classifier.component.html',
  styleUrls: ['./classifier.component.scss']
})
export class ClassifierComponent implements AfterViewInit {
  lat!: number;
  lon!: number;

  tileUrl: string | null = null;
  prediction: string | null = null;
  loading = false;
  error: string | null = null;

  private map!: L.Map;

  constructor(private svc: ClassificationService) {}

  ngAfterViewInit(): void {
    // Initialize Leaflet map when view is ready
    this.map = L.map('map', {
      maxBounds: L.latLngBounds([-90, -180], [90, 180]),
      maxBoundsViscosity: 1.0
    }).setView([36.8065, 10.1815], 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap'
    }).addTo(this.map);

    // Force map to recalculate its size
    setTimeout(() => this.map.invalidateSize(), 0);

    // Register click handler to capture lat/lon
    this.map.on('click', (e: L.LeafletMouseEvent) => {
      this.lat = e.latlng.lat;
      this.lon = e.latlng.lng;
    });

    // Prevent map from intercepting form button clicks
    const button = document.querySelector('.classifier-form button');
    if (button) {
      L.DomEvent.disableClickPropagation(button as HTMLElement);
    }
  }
 locateMe(): void {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        this.lat = pos.coords.latitude;
        this.lon = pos.coords.longitude;

        // Optionally pan the map to the user’s location:
        if (this.map) {
          this.map.setView([this.lat, this.lon], 13);
          // And place a marker:
          L.marker([this.lat, this.lon]).addTo(this.map)
            .bindPopup('You are here')
            .openPopup();
        }
      },
      (err) => {
        console.error('Geolocation error:', err);
        alert('Unable to retrieve your location');
      },
      { enableHighAccuracy: true, timeout: 5000 }
    );
  }

  onSubmit(): void {
    if (this.lat == null || this.lon == null) {
      this.error = 'Please select a location on the map first.';
      return;
    }

    this.reset();
    this.loading = true;

    this.svc.predict(this.lat, this.lon).subscribe({
      next: (res) => {
        this.tileUrl = `data:image/png;base64,${res.tile}`;
        this.prediction = res.prediction;
        this.loading = false;
      },
      error: () => {
        this.error = 'Failed to load tile or prediction.';
        this.loading = false;
      }
    });
  }

  private reset(): void {
    this.tileUrl = null;
    this.prediction = null;
    this.error = null;
  }
}
