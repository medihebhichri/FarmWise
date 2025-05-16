import { RouteInfo } from './sidebar.metadata';

export const ROUTES: RouteInfo[] = [
  {
    path: '/dashboard',
    title: 'Dashboard',
    icon: 'bi bi-speedometer2', // Represents speed and performance
    class: '',
    extralink: false,
    submenu: []
  },
  {
    path: '/classifier',
    title: 'Classifier',
    icon: 'bi bi-graph-up', // Represents data analysis and prediction
    class: '',
    extralink: false,
    submenu: []
  },
  {
    path: '/recommendation',
    title: 'Crop Recommendation',
    icon: 'bi bi-seedling', // Represents agriculture and crops
    class: '',
    extralink: false,
    submenu: []
  },
  {
    path: '/land',
    title: 'Land Price Prediction',
    icon: 'bi bi-cash-coin', // Represents financial predictions
    class: '',
    extralink: false,
    submenu: []
  },
  {
    path: '/about',
    title: 'About',
    icon: 'bi bi-info-circle', // Represents information
    class: '',
    extralink: false,
    submenu: []
  }
];
