import { ChartPositioningSystem } from '../chartPositioning';

describe('ChartPositioningSystem', () => {
  const defaultArea = {
    width: 1000,
    height: 800,
    margins: {
      top: 20,
      right: 20,
      bottom: 20,
      left: 20
    }
  };

  let system: ChartPositioningSystem;

  beforeEach(() => {
    system = new ChartPositioningSystem(defaultArea);
  });

  describe('Chart Addition and Removal', () => {
    test('should add chart successfully', () => {
      const chart = {
        position: { x: 100, y: 100 },
        dimensions: { width: 200, height: 150 }
      };

      system.addChart('chart1', chart);
      const positions = system.getChartPositions();
      expect(positions.get('chart1')).toBeDefined();
    });

    test('should remove chart successfully', () => {
      const chart = {
        position: { x: 100, y: 100 },
        dimensions: { width: 200, height: 150 }
      };

      system.addChart('chart1', chart);
      system.removeChart('chart1');
      const positions = system.getChartPositions();
      expect(positions.get('chart1')).toBeUndefined();
    });
  });

  describe('Chart Movement', () => {
    test('should move chart to valid position', () => {
      const chart = {
        position: { x: 100, y: 100 },
        dimensions: { width: 200, height: 150 }
      };

      system.addChart('chart1', chart);
      const result = system.moveChart('chart1', { x: 300, y: 200 });
      
      expect(result.available).toBe(true);
      expect(result.conflicts).toHaveLength(0);
    });

    test('should detect collision when moving', () => {
      const chart1 = {
        position: { x: 100, y: 100 },
        dimensions: { width: 200, height: 150 }
      };

      const chart2 = {
        position: { x: 400, y: 100 },
        dimensions: { width: 200, height: 150 }
      };

      system.addChart('chart1', chart1);
      system.addChart('chart2', chart2);

      const result = system.moveChart('chart1', { x: 350, y: 100 });
      
      expect(result.available).toBe(false);
      expect(result.conflicts).toHaveLength(1);
    });
  });

  describe('Space Finding', () => {
    test('should find available space for new chart', () => {
      const dimensions = { width: 200, height: 150 };
      const position = system.findAvailableSpace(dimensions);
      
      expect(position).toBeDefined();
      expect(position?.x).toBeGreaterThanOrEqual(defaultArea.margins.left);
      expect(position?.y).toBeGreaterThanOrEqual(defaultArea.margins.top);
    });

    test('should return null when no space available', () => {
      // Llenar el área con gráficos grandes
      const chart = {
        position: { x: 20, y: 20 },
        dimensions: { width: 960, height: 760 }
      };

      system.addChart('chart1', chart);
      
      const dimensions = { width: 200, height: 150 };
      const position = system.findAvailableSpace(dimensions);
      
      expect(position).toBeNull();
    });
  });

  describe('Optimal Distribution', () => {
    test('should distribute charts optimally', () => {
      const charts = [
        {
          position: { x: 100, y: 100 },
          dimensions: { width: 200, height: 150 }
        },
        {
          position: { x: 50, y: 50 },
          dimensions: { width: 150, height: 100 }
        },
        {
          position: { x: 300, y: 300 },
          dimensions: { width: 250, height: 200 }
        }
      ];

      charts.forEach((chart, index) => {
        system.addChart(`chart${index + 1}`, chart);
      });

      const distribution = system.getOptimalDistribution();
      
      expect(distribution.size).toBe(3);
      
      // Verificar que no hay superposiciones
      const positions = Array.from(distribution.values());
      for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          const bounds1 = positions[i];
          const bounds2 = positions[j];
          
          const hasCollision = (
            bounds1.position.x < bounds2.position.x + bounds2.dimensions.width &&
            bounds1.position.x + bounds1.dimensions.width > bounds2.position.x &&
            bounds1.position.y < bounds2.position.y + bounds2.dimensions.height &&
            bounds1.position.y + bounds1.dimensions.height > bounds2.position.y
          );
          
          expect(hasCollision).toBe(false);
        }
      }
    });
  });

  describe('Area Updates', () => {
    test('should handle area resize', () => {
      const chart = {
        position: { x: 800, y: 600 },
        dimensions: { width: 200, height: 150 }
      };

      system.addChart('chart1', chart);

      const newArea = {
        width: 800,
        height: 600,
        margins: defaultArea.margins
      };

      system.updateArea(newArea);
      const positions = system.getChartPositions();
      const updatedChart = positions.get('chart1');

      expect(updatedChart).toBeDefined();
      expect(updatedChart!.position.x + updatedChart!.dimensions.width)
        .toBeLessThanOrEqual(newArea.width - newArea.margins.right);
      expect(updatedChart!.position.y + updatedChart!.dimensions.height)
        .toBeLessThanOrEqual(newArea.height - newArea.margins.bottom);
    });
  });

  describe('Validation', () => {
    test('should throw error for invalid area', () => {
      expect(() => {
        new ChartPositioningSystem({
          ...defaultArea,
          width: -100
        });
      }).toThrow('Invalid area width');
    });

    test('should throw error for invalid chart bounds', () => {
      expect(() => {
        system.addChart('chart1', {
          position: { x: -100, y: 100 },
          dimensions: { width: 200, height: 150 }
        });
      }).toThrow('Invalid position');
    });
  });
});