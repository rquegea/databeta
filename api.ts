// src/Api.ts
import axios from 'axios';

// URL base de la API - ajusta esto a la URL correcta de tu backend
const API_BASE_URL = 'http://localhost:5001';

// Cliente axios configurado
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Importante si estás usando cookies/sesiones
});

// Funciones para interactuar con la API
export const chatApi = {
  // Enviar un mensaje al chatbot
  sendMessage: async (message: string) => {
    try {
      const response = await apiClient.post('/api/chat', { message });
      return response.data;
    } catch (error) {
      console.error('Error al enviar mensaje:', error);
      throw error;
    }
  },
  
  // Limpiar el historial de chat
  clearChat: async () => {
    try {
      const response = await apiClient.post('/api/clear-chat');
      return response.data;
    } catch (error) {
      console.error('Error al limpiar chat:', error);
      throw error;
    }
  }
};

export default apiClient;