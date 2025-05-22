import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictLandComponent } from './predict-land.component';

describe('PredictLandComponent', () => {
  let component: PredictLandComponent;
  let fixture: ComponentFixture<PredictLandComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [PredictLandComponent]
    });
    fixture = TestBed.createComponent(PredictLandComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
