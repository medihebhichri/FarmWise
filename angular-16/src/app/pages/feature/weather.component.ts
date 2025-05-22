import { Component, AfterViewInit } from '@angular/core';
import * as L from 'leaflet';
import { WeatherService } from '../../Services/weather.service';
import { CropService } from '../../Services/crop.service';

@Component({
  selector: 'app-weather',
  templateUrl: './weather.component.html',
  styleUrls: ['./weather.component.scss'],
})
export class WeatherComponent implements AfterViewInit {
  lat: number = 36.8;
  lon: number = 10.2;
  predictedCrop: string | null = null;
  weatherData: any = null;
  marker: L.Marker | null = null;
  loading: boolean = false; // Added loading property
  error: string | null = null; // Added error property

  soilData = {
    ph: 6.5,
    N: 0.3,
    P: 35,
    K: 50
  };

  constructor(
    private weatherService: WeatherService,
    private cropService: CropService
  ) {}

  ngAfterViewInit(): void {
    const map = L.map('map').setView([this.lat, this.lon], 6);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    map.on('click', (e: L.LeafletMouseEvent) => {
      this.lat = e.latlng.lat;
      this.lon = e.latlng.lng;

      if (this.marker) this.marker.remove();
      this.marker = L.marker([this.lat, this.lon]).addTo(map);

      this.fetchWeather(); // auto-fetch weather after selection
    });
  }

  fetchWeather() {
    this.loading = true; // Set loading to true when fetching weather
    this.weatherService.getWeather(this.lat, this.lon).subscribe({
      next: (data) => {
        this.weatherData = data;
        this.loading = false; // Set loading to false on successful data fetch
      },
      error: (err) => {
        console.error('Weather error:', err);
        this.error = 'Failed to load weather data.'; // Set error message
        this.loading = false; // Set loading to false on error
      }
    });
  }

  onSubmit() {
    if (!this.weatherData) {
      alert("Please click on the map to select a location and fetch weather.");
      return;
    }

    const input = {
      K: this.soilData.K,
      N: this.soilData.N,
      P: this.soilData.P,
      ph: this.soilData.ph,
      humidity: this.weatherData.humidity,
      temperature: this.weatherData.temperature,
      rainfall: this.weatherData.rainfall
    };

    this.cropService.predictCrop(input).subscribe({
      next: (res) => this.predictedCrop = res.recommended_crop,
      error: (err) => console.error('Prediction error:', err)
    });
  }
}
