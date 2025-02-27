import React, { useState, useEffect, useCallback, useRef } from 'react';
import { X, MoreVertical } from 'lucide-react';
import { 
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, BarChart, Bar, RadialBarChart, RadialBar
} from 'recharts';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// Mock data service
const fetchData = async (type: string) => {
  await new Promise(resolve => setTimeout(resolve, 500));
  
  switch (type) {
    case 'puntualidad_salida':
      return [
        { name: '6:00', valor: 98 }, { name: '7:00', valor: 96 },
        { name: '8:00', valor: 92 }, { name: '9:00', valor: 95 },
        { name: '10:00', valor: 97 }, { name: '11:00', valor: 99 },
        { name: '12:00', valor: 98 }
      ];
    case 'puntualidad_llegada':
      return [
        { name: '6:00', valor: 97 }, { name: '7:00', valor: 95 },
        { name: '8:00', valor: 93 }, { name: '9:00', valor: 94 },
        { name: '10:00', valor: 96 }, { name: '11:00', valor: 98 },
        { name: '12:00', valor: 97 }
      ];
    default:
      return [];
  }
};

const vuelosRetrasoData = {
  'taxi-in': [
    { name: 'A tiempo', value: 85 },
    { name: 'Retraso', value: 15 }
  ],
  'taxi-out': [
    { name: 'A tiempo', value: 80 },
    { name: 'Retraso', value: 20 }
  ]
};

const horasBloqueData = [
  { name: 'Lun', horas: 120 }, { name: 'Mar', horas: 125 },
  { name: 'Mie', horas: 118 }, { name: 'Jue', horas: 122 },
  { name: 'Vie', horas: 130 }, { name: 'Sab', horas: 115 },
  { name: 'Dom', horas: 110 }
];

const tiempoRotacionData = [
  { name: 'A320', tiempo: 45 },
  { name: 'A321', tiempo: 50 },
  { name: 'B737', tiempo: 40 },
  { name: 'B787', tiempo: 55 }
];

const tiempoRodajeData = {
  'taxi-in': [
    { name: 'A320', tiempo: 8 },
    { name: 'A321', tiempo: 9 },
    { name: 'B737', tiempo: 7 },
    { name: 'B787', tiempo: 10 }
  ],
  'taxi-out': [
    { name: 'A320', tiempo: 12 },
    { name: 'A321', tiempo: 13 },
    { name: 'B737', tiempo: 11 },
    { name: 'B787', tiempo: 14 }
  ]
};

const horasOperacionData = [
  { name: '6:00', horas: 10 }, { name: '8:00', horas: 25 },
  { name: '10:00', horas: 35 }, { name: '12:00', horas: 40 },
  { name: '14:00', horas: 38 }, { name: '16:00', horas: 30 },
  { name: '18:00', horas: 20 }
];

const analisisFlotaData = [
  { name: 'A320', value: 40 },
  { name: 'A321', value: 30 },
  { name: 'B737', value: 20 },
  { name: 'B787', value: 10 }
];

const utilizacionAeronavesData = [
  { name: 'Lun', A320: 12, A321: 11, B737: 10, B787: 9 },
  { name: 'Mar', A320: 11, A321: 12, B737: 11, B787: 10 },
  { name: 'Mie', A320: 13, A321: 10, B737: 12, B787: 11 },
  { name: 'Jue', A320: 12, A321: 13, B737: 11, B787: 10 },
  { name: 'Vie', A320: 14, A321: 12, B737: 13, B787: 12 },
  { name: 'Sab', A320: 11, A321: 10, B737: 9, B787: 8 },
  { name: 'Dom', A320: 10, A321: 9, B737: 8, B787: 7 }
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const visualizerOptions = [
  { id: 'puntualidad_salida', title: 'Puntualidad de Salida' },
  { id: 'puntualidad_llegada', title: 'Puntualidad de Llegada' },
  { id: 'vuelos_retraso', title: 'Vuelos con Retraso' },
  { id: 'horas_bloque', title: 'Horas de Bloque' },
  { id: 'tiempo_rotacion', title: 'Tiempo de Rotación' },
  { id: 'tiempo_rodaje', title: 'Tiempo de Rodaje' },
  { id: 'horas_operacion', title: 'Horas de Operación' },
  { id: 'analisis_flota', title: 'Análisis de Flota' },
  { id: 'utilizacion_aeronaves', title: 'Utilización de Aeronaves' }
];

interface OperationsProps {
  activeVisualizers: string[];
  animatingVisualizers: string[];
  onClose: (id: string) => void;
  onAddToPlayground?: (id: string) => void;
  layout?: GridLayout.Layout[];
  onLayoutChange?: (layout: GridLayout.Layout[]) => void;
}

const legendStyle = {
  fontSize: '9px',
  margin: 0,
  padding: '2px 4px',
  lineHeight: '1.2',
  maxWidth: '100%',
  overflowX: 'hidden',
  textOverflow: 'ellipsis',
  position: 'relative'
};

const tooltipStyle = {
  fontSize: '9px',
  padding: '4px 8px',
  backgroundColor: 'rgba(255, 255, 255, 0.98)',
  border: '1px solid #f0f0f0',
  borderRadius: '4px',
  boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
};

const axisStyle = {
  fontSize: 9,
  tickMargin: 4
};

const shouldShowLegend = (width: number, height: number) => {
  return width > 150 && height > 100;
};

const shouldShowAxisLabels = (width: number, height: number) => {
  return width > 120 && height > 80;
};

export function Operations({ 
  activeVisualizers, 
  animatingVisualizers, 
  onClose,
  onAddToPlayground,
  layout: propLayout,
  onLayoutChange
}: OperationsProps) {
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [chartData, setChartData] = useState<{ [key: string]: any[] }>({});
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<{ [key: string]: string }>({
    vuelos_retraso: 'taxi-in',
    tiempo_rodaje: 'taxi-in'
  });
  const [chartDimensions, setChartDimensions] = useState<{ [key: string]: { width: number; height: number } }>({});
  const menuRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [maxRows, setMaxRows] = useState(12);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.offsetWidth;
        const height = window.innerHeight - 144;
        const newMaxRows = Math.floor(height / 30);
        setContainerWidth(width);
        setMaxRows(newMaxRows);
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    const loadData = async (id: string) => {
      setLoading(prev => ({ ...prev, [id]: true }));
      try {
        const data = await fetchData(id);
        setChartData(prev => ({ ...prev, [id]: data }));
      } finally {
        setLoading(prev => ({ ...prev, [id]: false }));
      }
    };

    activeVisualizers.forEach(id => {
      if (!chartData[id]) {
        loadData(id);
      }
    });
  }, [activeVisualizers]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setActiveMenu(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLayoutChange = useCallback((newLayout: GridLayout.Layout[]) => {
    const adjustedLayout = newLayout.map(item => {
      if (item.y + item.h > maxRows) {
        return {
          ...item,
          y: Math.max(0, maxRows - item.h)
        };
      }
      return item;
    });

    onLayoutChange?.(adjustedLayout);
  }, [onLayoutChange, maxRows]);

  const updateChartDimensions = (id: string, width: number, height: number) => {
    setChartDimensions(prev => ({
      ...prev,
      [id]: { width, height }
    }));
  };

  const renderChart = (id: string) => {
    if (loading[id]) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      );
    }

    const dimensions = chartDimensions[id] || { width: 0, height: 0 };
    const showLegend = shouldShowLegend(dimensions.width, dimensions.height);
    const showAxisLabels = shouldShowAxisLabels(dimensions.width, dimensions.height);

    const commonProps = {
      margin: { 
        top: 5, 
        right: showLegend ? 10 : 5, 
        left: showAxisLabels ? -15 : -20, 
        bottom: 0 
      }
    };

    switch (id) {
      case 'puntualidad_salida':
      case 'puntualidad_llegada':
        return (
          <ResponsiveContainer 
            width="100%" 
            height="100%"
            onResize={(width, height) => updateChartDimensions(id, width, height)}
          >
            <LineChart data={chartData[id] || []} {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                {...axisStyle} 
                hide={!showAxisLabels}
                interval={dimensions.width < 200 ? 1 : 0}
              />
              <YAxis 
                domain={[90, 100]} 
                {...axisStyle} 
                width={20}
                hide={!showAxisLabels}
              />
              <Tooltip contentStyle={tooltipStyle} />
              {showLegend && (
                <Legend 
                  wrapperStyle={{
                    ...legendStyle,
                    bottom: -5,
                    left: '50%',
                    transform: 'translateX(-50%)'
                  }}
                  align="center"
                  verticalAlign="bottom"
                  height={20}
                />
              )}
              <Line 
                type="monotone" 
                dataKey="valor" 
                name="Punt. (%)" 
                stroke={id === 'puntualidad_salida' ? "#0088FE" : "#00C49F"}
                animationDuration={300}
                dot={{ r: dimensions.width < 200 ? 1 : 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'vuelos_retraso':
        return (
          <div>
            <div className="flex justify-center gap-2 mb-1">
              <button
                onClick={() => setActiveTab({ ...activeTab, vuelos_retraso: 'taxi-in' })}
                className={`text-[8px] px-2 py-0.5 rounded transition-colors duration-200 ${
                  activeTab.vuelos_retraso === 'taxi-in'
                    ? 'bg-blue-100 text-blue-600'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                Taxi-In
              </button>
              <button
                onClick={() => setActiveTab({ ...activeTab, vuelos_retraso: 'taxi-out' })}
                className={`text-[8px] px-2 py-0.5 rounded transition-colors duration-200 ${
                  activeTab.vuelos_retraso === 'taxi-out'
                    ? 'bg-blue-100 text-blue-600'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                Taxi-Out
              </button>
            </div>
            <ResponsiveContainer 
              width="100%" 
              height="90%"
              onResize={(width, height) => updateChartDimensions(id, width, height)}
            >
              <PieChart margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                <Pie
                  data={vuelosRetrasoData[activeTab.vuelos_retraso]}
                  cx="50%"
                  cy="50%"
                  innerRadius={dimensions.width < 200 ? 25 : 35}
                  outerRadius={dimensions.width < 200 ? 40 : 50}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                  label={dimensions.width > 150 ? ({ name, value }) => `${name.substring(0, 1)}: ${value}%` : false}
                  labelLine={false}
                  animationDuration={300}
                >
                  {vuelosRetrasoData[activeTab.vuelos_retraso].map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
                {showLegend && (
                  <Legend 
                    wrapperStyle={legendStyle}
                    formatter={(value) => value === 'A tiempo' ? 'Ok' : 'Ret'}
                  />
                )}
              </PieChart>
            </ResponsiveContainer>
          </div>
        );

      case 'horas_bloque':
        return (
          <ResponsiveContainer 
            width="100%" 
            height="100%"
            onResize={(width, height) => updateChartDimensions(id, width, height)}
          >
            <LineChart data={horasBloqueData} {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                {...axisStyle}
                hide={!showAxisLabels}
              />
              <YAxis 
                {...axisStyle} 
                width={20}
                hide={!showAxisLabels}
              />
              <Tooltip contentStyle={tooltipStyle} />
              {showLegend && (
                <Legend 
                  wrapperStyle={legendStyle}
                  formatter={(value) => value === 'horas' ? 'Hrs' : value}
                />
              )}
              <Line 
                type="monotone" 
                dataKey="horas" 
                name="Hrs" 
                stroke="#FFBB28"
                animationDuration={300}
                dot={{ r: dimensions.width < 200 ? 1 : 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'tiempo_rotacion':
        return (
          <ResponsiveContainer 
            width="100%" 
            height="100%"
            onResize={(width, height) => updateChartDimensions(id, width, height)}
          >
            <BarChart data={tiempoRotacionData} {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                {...axisStyle}
                hide={!showAxisLabels}
              />
              <YAxis 
                {...axisStyle} 
                width={20}
                hide={!showAxisLabels}
              />
              <Tooltip contentStyle={tooltipStyle} />
              {showLegend && (
                <Legend 
                  wrapperStyle={legendStyle}
                  formatter={(value) => value === 'tiempo' ? 'Min' : value}
                />
              )}
              <Bar 
                dataKey="tiempo" 
                name="Min" 
                fill="#FF8042"
                animationDuration={300}
              />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'tiempo_rodaje':
        return (
          <div>
            <div className="flex justify-center gap-2 mb-1">
              <button
                onClick={() => setActiveTab({ ...activeTab, tiempo_rodaje: 'taxi-in' })}
                className={`text-[8px] px-2 py-0.5 rounded transition-colors duration-200 ${
                  activeTab.tiempo_rodaje === 'taxi-in'
                    ? 'bg-blue-100 text-blue-600'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                Taxi-In
              </button>
              <button
                onClick={() => setActiveTab({ ...activeTab, tiempo_rodaje: 'taxi-out' })}
                className={`text-[8px] px-2 py-0.5 rounded transition-colors duration-200 ${
                  activeTab.tiempo_rodaje === 'taxi-out'
                    ? 'bg-blue-100 text-blue-600'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                Taxi-Out
              </button>
            </div>
            <ResponsiveContainer 
              width="100%" 
              height="90%"
              onResize={(width, height) => updateChartDimensions(id, width, height)}
            >
              <BarChart data={tiempoRodajeData[activeTab.tiempo_rodaje]} {...commonProps}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="name" 
                  {...axisStyle}
                  hide={!showAxisLabels}
                />
                <YAxis 
                  {...axisStyle} 
                  width={20}
                  hide={!showAxisLabels}
                />
                <Tooltip contentStyle={tooltipStyle} />
                {showLegend && (
                  <Legend 
                    wrapperStyle={legendStyle}
                    formatter={(value) => value === 'tiempo' ? 'Min' : value}
                  />
                )}
                <Bar 
                  dataKey="tiempo" 
                  name="Min" 
                  fill="#8884d8"
                  animationDuration={300}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );

      case 'horas_operacion':
        return (
          <ResponsiveContainer 
            width="100%" 
            height="100%"
            onResize={(width, height) => updateChartDimensions(id, width, height)}
          >
            <LineChart data={horasOperacionData} {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                {...axisStyle}
                hide={!showAxisLabels}
              />
              <YAxis 
                {...axisStyle} 
                width={20}
                hide={!showAxisLabels}
              />
              <Tooltip contentStyle={tooltipStyle} />
              {showLegend && (
                <Legend 
                  wrapperStyle={legendStyle}
                  formatter={(value) => value === 'horas' ? 'Hrs' : value}
                />
              )}
              <Line 
                type="monotone" 
                dataKey="horas" 
                name="Hrs" 
                stroke="#0088FE"
                animationDuration={300}
                dot={{ r: dimensions.width < 200 ? 1 : 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'analisis_flota':
        return (
          <ResponsiveContainer 
            width="100%" 
            height="100%"
            onResize={(width, height) => updateChartDimensions(id, width, height)}
          >
            <RadialBarChart
              innerRadius="30%"
              outerRadius="80%"
              data={analisisFlotaData}
              startAngle={180}
              endAngle={0}
              margin={{ top: 5, right: 5, bottom: 5, left: 5 }}
              animationDuration={300}
            >
              <RadialBar
                minAngle={15}
                label={dimensions.width > 150 ? { fill: '#666', fontSize: 8, position: 'insideStart' } : false}
                background
                clockWise={true}
                dataKey="value"
              >
                {analisisFlotaData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </RadialBar>
              <Tooltip contentStyle={tooltipStyle} />
              {showLegend && (
                <Legend 
                  wrapperStyle={legendStyle}
                  iconSize={8}
                  formatter={(value) => value}
                />
              )}
            </RadialBarChart>
          </ResponsiveContainer>
        );

      case 'utilizacion_aeronaves':
        return (
          <ResponsiveContainer 
            width="100%" 
            height="100%"
            onResize={(width, height) => updateChartDimensions(id, width, height)}
          >
            <LineChart data={utilizacionAeronavesData} {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                {...axisStyle}
                hide={!showAxisLabels}
              />
              <YAxis 
                {...axisStyle} 
                width={20}
                hide={!showAxisLabels}
              />
              <Tooltip contentStyle={tooltipStyle} />
              {showLegend && (
                <Legend 
                  wrapperStyle={legendStyle}
                  iconSize={8}
                  formatter={(value) => value}
                />
              )}
              {Object.keys(utilizacionAeronavesData[0])
                .filter(key => key !== 'name')
                .map((key, index) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={COLORS[index % COLORS.length]}
                    strokeWidth={1}
                    dot={{ r: dimensions.width < 200 ? 1 : 1.5 }}
                    animationDuration={300}
                  />
                ))}
            </LineChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  return (
    <div className="px-2 py-4 pb-24" ref={containerRef}>
      <div className="grid-container">
        <GridLayout
          className="layout"
          layout={propLayout || []}
          cols={24}
          rowHeight={30}
          width={containerWidth}
          isDraggable={true}
          isResizable={true}
          margin={[8, 8]}
          containerPadding={[0, 0]}
          preventCollision={false}
          compactType={null}
          maxRows={maxRows}
          onLayoutChange={handleLayoutChange}
          resizeHandles={['se', 'sw', 'ne', 'nw']}
          draggableCancel=".non-draggable"
        >
          {activeVisualizers.map((id) => (
            <div 
              key={id} 
              className={`bg-white rounded-lg shadow-sm p-2 transition-all duration-300 
                hover:shadow-lg hover:bg-white/80 group ${
                animatingVisualizers.includes(id) ? 'smoke-in' : ''
              }`}
            >
              <div className="flex justify-between items-center mb-2 non-draggable">
                <h2 className="text-[11px] font-semibold text-gray-700 px-2.5 py-1 bg-gray-50/80 group-hover:bg-gray-100/90 rounded-full transition-colors duration-200">
                  {visualizerOptions.find(v => v.id === id)?.title}
                </h2>
                <div className="flex items-center gap-1.5">
                  {onAddToPlayground && (
                    <div className="relative" ref={menuRef}>
                      <button
                        onClick={() => setActiveMenu(activeMenu === id ? null : id)}
                        className="w-5 h-5 flex items-center justify-center rounded-full 
                                 text-gray-400 hover:text-gray-600 hover:bg-gray-100
                                 transition-colors duration-200"
                      >
                        <MoreVertical className="w-3 h-3" />
                      </button>
                      {activeMenu === id && (
                        <div className="absolute right-0 mt-1 w-36 bg-white rounded-lg shadow-lg py-1 z-50">
                          <button
                            onClick={() => {
                              onAddToPlayground(id);
                              setActiveMenu(null);
                            }}
                            className="w-full text-left px-3 py-1.5 text-[10px] text-gray-600 hover:bg-gray-50"
                          >
                            Añadir al Playground
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  <button
                    onClick={() => onClose(id)}
                    className="w-5 h-5 flex items-center justify-center rounded-full 
                             text-gray-400 hover:text-gray-600 hover:bg-gray-100
                             transition-colors duration-200"
                    aria-label="Cerrar gráfico"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              </div>
              <div className="h-[calc(100%-2.25rem)] non-draggable">
                {renderChart(id)}
              </div>
            </div>
          ))}
        </GridLayout>
      </div>
    </div>
  );
}