import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { FullComponent } from './layouts/full/full.component';
import { LoginComponent } from './pages/auth/login/login.component';
import { SignupComponent } from './pages/auth/signup/signup.component';
import { ProfileComponent } from './pages/profile/profile.component';
import { ClassifierComponent } from './pages/classifier/classifier.component';
import { WeatherComponent } from './pages/feature/weather.component';
import { PredictLandComponent } from './pages/predict-land/predict-land.component';

export const Approutes: Routes = [
  {
    path: '',
    component: FullComponent,
    children: [
      {
        path: 'dashboard',
        loadChildren: () => import('./dashboard/dashboard.module').then(m => m.DashboardModule)
      },
      {
        path: 'about',
        loadChildren: () => import('./about/about.module').then(m => m.AboutModule)
      },
      {
        path: 'component',
        loadChildren: () => import('./component/component.module').then(m => m.ComponentsModule)
      },
      {
        path: 'profile',
        component: ProfileComponent
      },
        {
    path: 'classifier',
    component: ClassifierComponent
  },
    {
    path: 'recommendation',
    component: WeatherComponent
  },
      {
        path: 'land',
        component: PredictLandComponent
      },

    ]},
  {
    path: 'auth',
    children: [
      { path: 'login', component: LoginComponent },
      { path: 'signup', component: SignupComponent },
      
    ]
  },
 

  
  { path: '**', redirectTo: 'auth/login' }
];
