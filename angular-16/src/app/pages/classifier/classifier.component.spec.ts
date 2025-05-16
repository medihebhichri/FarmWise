import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassifierComponent } from './classifier.component';

describe('ClassifierComponent', () => {
  let component: ClassifierComponent;
  let fixture: ComponentFixture<ClassifierComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ClassifierComponent]
    });
    fixture = TestBed.createComponent(ClassifierComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
