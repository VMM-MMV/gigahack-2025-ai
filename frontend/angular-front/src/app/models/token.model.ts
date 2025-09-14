export interface TokenData {
  key: string;
  value: string;
}

export interface ApiResponse {
  data?: any[];
  message?: string;
  status?: string;
}

export interface AppConfig {
  apiBaseUrl: string;
  defaultEndpoint: string;
  refreshInterval?: number;
}
