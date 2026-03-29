import axios from "axios";
import { BACKEND_URL } from "../constants/config";
import { useConnectionStore } from "../stores/useConnectionStore";
import { getToken, removeToken, removeUser } from "../utils/storage";

const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30000,
});

// Dynamically set baseURL from connection store on every request
api.interceptors.request.use(async (config) => {
  config.baseURL = useConnectionStore.getState().backendUrl;
  const token = await getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      await removeToken();
      await removeUser();
    }
    return Promise.reject(error);
  }
);

export default api;
