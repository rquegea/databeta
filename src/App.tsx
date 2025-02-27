import React, { useState, useEffect, useRef } from 'react';
import { Send, ChevronDown, X } from 'lucide-react';
import { Layout } from './Layout';
import { Operations } from './components/Operations';

interface NavItem {
  label: string;
  path: string;
  submenu?: {
    label: string;
    id: string;
  }[];
}

const navItems: NavItem[] = [
  { 
    label: 'Playground', 
    path: '/playground' 
  },
  { 
    label: 'Operaciones', 
    path: '/operations',
    submenu: [
      { label: 'Puntualidad de Salida', id: 'puntualidad_salida' },
      { label: 'Puntualidad de Llegada', id: 'puntualidad_llegada' },
      { label: 'Vuelos con Retraso', id: 'vuelos_retraso' },
      { label: 'Horas de Bloque', id: 'horas_bloque' },
      { label: 'Tiempo de Rotación', id: 'tiempo_rotacion' },
      { label: 'Tiempo de Rodaje', id: 'tiempo_rodaje' },
      { label: 'Horas de Operación', id: 'horas_operacion' },
      { label: 'Análisis de Flota', id: 'analisis_flota' },
      { label: 'Utilización de Aeronaves', id: 'utilizacion_aeronaves' }
    ]
  },
  { 
    label: 'Red y Flota', 
    path: '/network' 
  },
  { 
    label: 'Configuración', 
    path: '/settings' 
  },
];

interface ChartLayout {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

// Interfaz para los mensajes del chat
interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

function App() {
  const [message, setMessage] = useState('');
  const [isInputVisible, setIsInputVisible] = useState(false);
  const [activeNav, setActiveNav] = useState('/operations');
  const [activeSubmenu, setActiveSubmenu] = useState<string | null>(null);
  const [operationsVisualizers, setOperationsVisualizers] = useState<string[]>([]);
  const [playgroundVisualizers, setPlaygroundVisualizers] = useState<string[]>([]);
  const [animatingVisualizers, setAnimatingVisualizers] = useState<string[]>([]);
  const [layouts, setLayouts] = useState<{ [key: string]: ChartLayout[] }>({
    operations: [],
    playground: []
  });
  
  // Estados para el chat
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  
  const menuRef = useRef<HTMLDivElement>(null);
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const maxRows = 8;

  const findEmptySpace = (currentLayouts: ChartLayout[]): { x: number; y: number } | null => {
    const grid = Array(maxRows).fill(0).map(() => Array(24).fill(false));
    
    // Marcar espacios ocupados
    currentLayouts.forEach(layout => {
      for (let y = layout.y; y < layout.y + layout.h; y++) {
        for (let x = layout.x; x < layout.x + layout.w; x++) {
          if (y >= 0 && y < maxRows && x >= 0 && x < 24) {
            grid[y][x] = true;
          }
        }
      }
    });

    // Buscar espacio libre para un nuevo gráfico (12x6)
    for (let y = 0; y < maxRows - 5; y++) {
      for (let x = 0; x < 24 - 11; x++) {
        let isFree = true;
        
        // Verificar área 12x6
        for (let dy = 0; dy < 6; dy++) {
          for (let dx = 0; dx < 12; dx++) {
            if (grid[y + dy][x + dx]) {
              isFree = false;
              break;
            }
          }
          if (!isFree) break;
        }
        
        if (isFree) {
          return { x, y };
        }
      }
    }
    
    return null;
  };

  const hasAvailableSpace = (currentLayouts: ChartLayout[]): boolean => {
    // Verificar si hay espacio libre
    const emptySpace = findEmptySpace(currentLayouts);
    if (emptySpace) return true;

    // Si no hay espacio libre, verificar si hay espacio después del último gráfico
    if (currentLayouts.length === 0) return true;

    const maxY = Math.max(...currentLayouts.map(layout => layout.y + layout.h));
    return maxY + 6 <= maxRows;
  };

  const findOptimalPosition = (currentLayouts: ChartLayout[]): { x: number; y: number } => {
    // Intentar encontrar un espacio vacío primero
    const emptySpace = findEmptySpace(currentLayouts);
    if (emptySpace) return emptySpace;

    // Si no hay espacio vacío, colocar después del último gráfico
    if (currentLayouts.length === 0) return { x: 0, y: 0 };

    const maxY = Math.max(...currentLayouts.map(layout => layout.y + layout.h));
    return { x: 0, y: Math.min(maxY, maxRows - 6) };
  };

  const showSpaceWarning = () => {
    const warning = document.createElement('div');
    warning.className = 'fixed top-4 right-4 bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded shadow-lg z-50 animate-fade-in';
    warning.innerHTML = `
      <div class="flex items-center">
        <div class="py-1"><svg class="fill-current h-6 w-6 text-red-500 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zm12.73-1.41A8 8 0 1 0 4.34 4.34a8 8 0 0 0 11.32 11.32zM9 11V9h2v6H9v-4zm0-6h2v2H9V5z"/></svg></div>
        <div>
          <p class="font-bold">No hay espacio suficiente</p>
          <p class="text-sm">Por favor, reorganiza o elimina algunos gráficos para añadir uno nuevo.</p>
        </div>
      </div>
    `;
    document.body.appendChild(warning);
    setTimeout(() => {
      warning.classList.add('animate-fade-out');
      setTimeout(() => warning.remove(), 300);
    }, 3000);
  };

  const toggleVisualizer = (id: string) => {
    if (activeNav === '/operations') {
      if (operationsVisualizers.includes(id)) {
        setOperationsVisualizers(prev => prev.filter(v => v !== id));
        setLayouts(prev => ({
          ...prev,
          operations: prev.operations.filter(layout => layout.i !== id)
        }));
      } else {
        // Verificar espacio disponible considerando el layout actual
        if (!hasAvailableSpace(layouts.operations)) {
          showSpaceWarning();
          return;
        }

        const position = findOptimalPosition(layouts.operations);
        
        setOperationsVisualizers(prev => [...prev, id]);
        setAnimatingVisualizers(prev => [...prev, id]);
        
        setLayouts(prev => ({
          ...prev,
          operations: [
            ...prev.operations,
            {
              i: id,
              x: position.x,
              y: position.y,
              w: 12,
              h: 6
            }
          ]
        }));

        setTimeout(() => {
          setAnimatingVisualizers(prev => prev.filter(v => v !== id));
        }, 300);
      }
    }
  };

  const addToPlayground = (id: string) => {
    if (!playgroundVisualizers.includes(id)) {
      // Verificar espacio disponible en el playground
      if (!hasAvailableSpace(layouts.playground)) {
        showSpaceWarning();
        return;
      }

      const position = findOptimalPosition(layouts.playground);

      setPlaygroundVisualizers(prev => [...prev, id]);
      setLayouts(prev => ({
        ...prev,
        playground: [
          ...prev.playground,
          {
            i: id,
            x: position.x,
            y: position.y,
            w: 12,
            h: 6
          }
        ]
      }));
      
      if (activeNav === '/operations') {
        setActiveNav('/playground');
      }
    }
  };

  const removeFromPlayground = (id: string) => {
    setPlaygroundVisualizers(prev => prev.filter(v => v !== id));
    setLayouts(prev => ({
      ...prev,
      playground: prev.playground.filter(item => item.i !== id)
    }));
  };

  // Función actualizada para manejar el envío de mensajes al backend
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim()) return;
    
    const userMessage = message;
    setChatHistory(prev => [...prev, { role: 'user', content: userMessage }]);
    setMessage('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });
      
      const data = await response.json();
      
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: data.response || 'No se pudo procesar la respuesta' 
      }]);
    } catch (error) {
      console.error('Error:', error);
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: 'Error de conexión. Por favor, verifica el servidor.' 
      }]);
    } finally {
      setIsLoading(false);
      setShowChat(true);
    }
  };

  const handleNavClick = (item: NavItem) => {
    setActiveNav(item.path);
    if (item.submenu) {
      setActiveSubmenu(activeSubmenu === item.path ? null : item.path);
    } else {
      setActiveSubmenu(null);
    }
  };

  const handleChatAreaMouseEnter = () => {
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
    }
    hoverTimerRef.current = setTimeout(() => {
      setIsInputVisible(true);
    }, 800);
  };

  const handleChatAreaMouseLeave = () => {
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
    }
    if (!chatAreaRef.current?.contains(document.activeElement)) {
      setIsInputVisible(false);
    }
  };

  // Efecto para desplazar el historial de chat al último mensaje
  useEffect(() => {
    if (chatHistoryRef.current && chatHistory.length > 0) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setActiveSubmenu(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);

  const handleLayoutChange = (newLayout: ChartLayout[], view: 'operations' | 'playground') => {
    // Validar que ningún gráfico exceda los límites
    const isValidLayout = newLayout.every(item => {
      const bottomEdge = item.y + item.h;
      const rightEdge = item.x + item.w;
      return bottomEdge <= maxRows && rightEdge <= 24;
    });

    if (isValidLayout) {
      setLayouts(prev => ({
        ...prev,
        [view]: newLayout
      }));
    }
  };

  // Componente para mostrar el historial de chat
  const renderChatHistory = () => {
    if (chatHistory.length === 0 && !showChat) return null;
    
    return (
      <div 
        ref={chatHistoryRef}
        className={`fixed bottom-24 left-0 w-full px-2 z-30 max-h-[60vh] overflow-y-auto
                   transition-all duration-300 ease-in-out
                   ${showChat ? 'opacity-100' : 'opacity-0'}`}
      >
        <div className="max-w-xl mx-auto space-y-2 pb-4">
          {chatHistory.map((msg, index) => (
            <div 
              key={index}
              className={`p-2 rounded-lg max-w-[85%] ${
                msg.role === 'user' 
                  ? 'bg-blue-100 text-blue-800 ml-auto'
                  : 'bg-white/90 backdrop-blur-sm shadow-sm'
              }`}
            >
              <p className="text-[12px]">{msg.content}</p>
            </div>
          ))}
          {isLoading && (
            <div className="bg-white/90 backdrop-blur-sm shadow-sm p-2 rounded-lg animate-pulse">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <Layout>
      <div className="flex flex-col h-screen">
        <nav className="w-full h-12 bg-white/80 backdrop-blur-sm shadow-sm fixed top-0 left-0 z-50">
          <div className="container mx-auto px-2">
            <div className="h-12 flex items-center justify-between">
              <h1 className="text-[10px] font-bold text-gray-800 tracking-wider">TORGOS</h1>

              <div className="flex items-center space-x-4" ref={menuRef}>
                {navItems.map((item) => (
                  <div key={item.path} className="relative">
                    <button
                      onClick={() => handleNavClick(item)}
                      className={`text-[10px] transition-colors duration-200 flex items-center gap-1 ${
                        activeNav === item.path
                          ? 'text-blue-600 font-medium'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      {item.label}
                      {item.submenu && (
                        <ChevronDown 
                          className={`w-3 h-3 transition-transform duration-200 ${
                            activeSubmenu === item.path ? 'rotate-180' : ''
                          }`}
                        />
                      )}
                    </button>

                    {item.submenu && activeSubmenu === item.path && (
                      <div 
                        className="absolute top-full right-0 mt-1 w-48 bg-white rounded-lg shadow-lg py-1 z-50
                                 transform transition-all duration-200 ease-out origin-top-right
                                 opacity-100 scale-100"
                      >
                        {item.submenu.map((subitem) => (
                          <button
                            key={subitem.id}
                            onClick={() => toggleVisualizer(subitem.id)}
                            className={`w-full text-left px-3 py-1.5 text-[10px] flex items-center justify-between hover:bg-gray-50 ${
                              operationsVisualizers.includes(subitem.id)
                                ? 'text-blue-600'
                                : 'text-gray-600'
                            }`}
                          >
                            <span>{subitem.label}</span>
                            <div 
                              className={`ios-switch ${operationsVisualizers.includes(subitem.id) ? 'active' : ''}`}
                              role="switch"
                              aria-checked={operationsVisualizers.includes(subitem.id)}
                            >
                              <div className="ios-switch-handle" />
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </nav>

        <div className="flex-1 overflow-hidden mt-12">
          {activeNav === '/operations' && (
            <Operations 
              activeVisualizers={operationsVisualizers}
              animatingVisualizers={animatingVisualizers}
              onClose={toggleVisualizer}
              onAddToPlayground={addToPlayground}
              layout={layouts.operations}
              onLayoutChange={(newLayout) => handleLayoutChange(newLayout, 'operations')}
            />
          )}
          {activeNav === '/playground' && (
            <Operations 
              activeVisualizers={playgroundVisualizers}
              animatingVisualizers={animatingVisualizers}
              onClose={removeFromPlayground}
              layout={layouts.playground}
              onLayoutChange={(newLayout) => handleLayoutChange(newLayout, 'playground')}
            />
          )}
        </div>

        {/* Renderizar el historial de chat */}
        {renderChatHistory()}

        <div 
          ref={chatAreaRef}
          className="fixed bottom-0 left-0 w-full h-24 px-2 z-40"
          onMouseEnter={handleChatAreaMouseEnter}
          onMouseLeave={handleChatAreaMouseLeave}
        >
          <form 
            onSubmit={handleSubmit}
            className={`max-w-xl mx-auto transition-all duration-300 ease-in-out ${
              isInputVisible 
                ? 'opacity-100 transform translate-y-0' 
                : 'opacity-0 transform translate-y-24'
            }`}
          >
            <div className="relative flex items-center group">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Escribe un mensaje..."
                className="w-full h-[35px] pl-4 pr-10 rounded-full bg-white/95 
                          shadow-lg backdrop-blur-sm
                          focus:outline-none focus:ring-1 focus:ring-blue-500 
                          transition-all duration-300 ease-in-out
                          group-hover:h-[45px]
                          text-[12px] placeholder-gray-400"
                aria-label="Mensaje de chat"
                onFocus={() => {
                  setIsInputVisible(true);
                  setShowChat(true);
                }}
                onBlur={(e) => {
                  if (!chatAreaRef.current?.contains(e.relatedTarget)) {
                    setIsInputVisible(false);
                  }
                }}
              />
              <button
                type="submit"
                className="absolute right-2 w-7 h-7 flex items-center justify-center 
                         bg-blue-500 text-white rounded-full hover:bg-blue-600 
                         focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-blue-500
                         transition-all duration-300 ease-in-out
                         group-hover:w-9 group-hover:h-9
                         disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!message.trim() || isLoading}
                aria-label="Enviar mensaje"
              >
                <Send className="w-3 h-3 group-hover:w-4 group-hover:h-4" />
              </button>
            </div>
          </form>
        </div>
      </div>
    </Layout>
  );
}

export default App;