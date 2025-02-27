import { useRef, useEffect, useState, useCallback } from 'react';
import { ChartPositioningSystem } from '../utils/chartPositioning';

interface UseChartPositioningProps {
  containerRef: React.RefObject<HTMLElement>;
  initialCharts?: Map<string, {
    position: { x: number; y: number };
    dimensions: { width: number; height: number };
  }>;
}

export function useChartPositioning({ containerRef, initialCharts }: UseChartPositioningProps) {
  const systemRef = useRef<ChartPositioningSystem | null>(null);
  const [charts, setCharts] = useState(initialCharts || new Map());
  const [isDragging, setIsDragging] = useState(false);
  const [dragFeedback, setDragFeedback] = useState<{
    available: boolean;
    conflicts: any[];
  } | null>(null);

  // Inicializar el sistema cuando el contenedor está disponible
  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      systemRef.current = new ChartPositioningSystem({
        width: document.documentElement.scrollWidth,
        height: document.documentElement.scrollHeight,
        margins: {
          top: 0,
          right: 0,
          bottom: 0,
          left: 0
        }
      });

      // Inicializar con los gráficos existentes
      initialCharts?.forEach((chart, id) => {
        systemRef.current?.addChart(id, chart);
      });

      setCharts(systemRef.current.getChartPositions());
    }
  }, [containerRef, initialCharts]);

  // Actualizar el área cuando cambia el tamaño del contenedor
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && systemRef.current) {
        systemRef.current.updateArea({
          width: document.documentElement.scrollWidth,
          height: document.documentElement.scrollHeight,
          margins: {
            top: 0,
            right: 0,
            bottom: 0,
            left: 0
          }
        });
        setCharts(systemRef.current.getChartPositions());
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [containerRef]);

  const addChart = useCallback((id: string, chart: {
    position: { x: number; y: number };
    dimensions: { width: number; height: number };
  }) => {
    if (systemRef.current) {
      systemRef.current.addChart(id, chart);
      setCharts(systemRef.current.getChartPositions());
    }
  }, []);

  const removeChart = useCallback((id: string) => {
    if (systemRef.current) {
      systemRef.current.removeChart(id);
      setCharts(systemRef.current.getChartPositions());
    }
  }, []);

  const moveChart = useCallback((id: string, newPosition: { x: number; y: number }) => {
    if (systemRef.current) {
      const result = systemRef.current.moveChart(id, newPosition);
      setDragFeedback(result);
      if (result.available) {
        setCharts(systemRef.current.getChartPositions());
      }
      return result;
    }
    return { available: false, conflicts: [] };
  }, []);

  const resizeChart = useCallback((id: string, newDimensions: { width: number; height: number }) => {
    if (systemRef.current) {
      const result = systemRef.current.resizeChart(id, newDimensions);
      if (result.available) {
        setCharts(systemRef.current.getChartPositions());
      }
      return result;
    }
    return { available: false, conflicts: [] };
  }, []);

  const optimizeLayout = useCallback(() => {
    if (systemRef.current) {
      const newPositions = systemRef.current.getOptimalDistribution();
      newPositions.forEach((chart, id) => {
        systemRef.current?.moveChart(id, chart.position);
      });
      setCharts(systemRef.current.getChartPositions());
    }
  }, []);

  const findAvailableSpace = useCallback((dimensions: { width: number; height: number }) => {
    if (systemRef.current) {
      return systemRef.current.findAvailableSpace(dimensions);
    }
    return null;
  }, []);

  return {
    charts,
    isDragging,
    dragFeedback,
    addChart,
    removeChart,
    moveChart,
    resizeChart,
    optimizeLayout,
    findAvailableSpace,
    setIsDragging
  };
}