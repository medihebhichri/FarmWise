// src/app/predict-land/predict-land.component.ts
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';

import { switchMap }         from 'rxjs/operators';
import { CurrencyService } from 'src/app/Services/currency.service';
import { PredictionService } from 'src/app/Services/prediction.service';

@Component({
  selector: 'app-predict-land',
  templateUrl: './predict-land.component.html',
  styleUrls: ['./predict-land.component.scss']
})
export class PredictLandComponent {
  year!: number;
  stateEncoded!: number;
  landUseEncoded!: number;

  predictedUSD: number | null = null;
  predictedTND: number | null = null;
  predicting    = false;
  loadingRate   = false;
  rateError     = false; // Added rateError property

  private inputDim = 25;

  constructor(
    private predictionService: PredictionService,
    private currencyService:   CurrencyService
  ) {}

  onSubmit(form: NgForm) {
    if (form.invalid) return;

    // Reset everything
    this.predictedUSD = null;
    this.predictedTND = null;
    this.predicting   = true;
    this.loadingRate  = false;

    // Build your feature vector
    const zeros = Array(this.inputDim - 3).fill(0);
    const features = [ this.year, this.stateEncoded, this.landUseEncoded, ...zeros ];

    // 1) Predict USD, then 2) fetch rate, then compute TND
    this.predictionService.predict(features).pipe(
      switchMap(({ predicted_price }) => {
        this.predictedUSD = predicted_price;
        this.loadingRate  = true;
        return this.currencyService.getUsdToTndRate();
      })
    )
    .subscribe({
      next: rate => {
        this.predictedTND = this.predictedUSD! * rate;
        this.loadingRate  = false;
        this.predicting   = false;
      },
      error: err => {
        console.error('Error in prediction or rate fetch', err);
        this.loadingRate  = false;
        this.predicting   = false;
      }
    });
  }
}
