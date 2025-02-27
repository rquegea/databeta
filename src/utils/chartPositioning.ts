interface Position {
  x: number;
  y: number;
}

interface Dimensions {
  width: number;
  height: number;
}

interface Area {
  width: number;
  height: number;
  margins: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

interface ChartBounds {
  position: Position;
  dimensions: Dimensions;
}

interface SpaceResult {
  available: boolean;
  position?: Position;
  conflicts: ChartBounds[];
  availableSpaces: AvailableSpace[];
}

interface AvailableSpace {
  position: Position;
  dimensions: Dimensions;
  score: number; // Puntuación basada en la ubicación y tamaño
}

export class ChartPositioningSystem {
  private area: Area;
  private charts: Map<string, ChartBounds>;
  private gridSize = 8;
  private spaceGrid: boolean[][] = [];
  private availableSpaces: AvailableSpace[] = [];

  constructor(area: Area) {
    this.validateArea(area);
    this.area = area;
    this.charts = new Map();
    this.initializeSpaceGrid();
  }

  private initializeSpaceGrid(): void {
    const rows = Math.ceil(this.area.height / this.gridSize);
    const cols = Math.ceil(this.area.width / this.gridSize);
    this.spaceGrid = Array(rows).fill(false).map(() => Array(cols).fill(true));
  }

  private updateSpaceGrid(): void {
    this.initializeSpaceGrid();
    
    // Marcar celdas ocupadas por gráficos
    this.charts.forEach(chart => {
      const startX = Math.floor(chart.position.x / this.gridSize);
      const startY = Math.floor(chart.position.y / this.gridSize);
      const endX = Math.ceil((chart.position.x + chart.dimensions.width) / this.gridSize);
      const endY = Math.ceil((chart.position.y + chart.dimensions.height) / this.gridSize);

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          if (y >= 0 && y < this.spaceGrid.length && x >= 0 && x < this.spaceGrid[0].length) {
            this.spaceGrid[y][x] = false;
          }
        }
      }
    });

    // Encontrar espacios disponibles
    this.findAvailableSpaces();
  }

  private findAvailableSpaces(): void {
    this.availableSpaces = [];
    const visited = Array(this.spaceGrid.length).fill(false)
      .map(() => Array(this.spaceGrid[0].length).fill(false));

    for (let y = 0; y < this.spaceGrid.length; y++) {
      for (let x = 0; x < this.spaceGrid[0].length; x++) {
        if (this.spaceGrid[y][x] && !visited[y][x]) {
          const space = this.expandSpace(x, y, visited);
          if (space) {
            // Calcular puntuación basada en ubicación y tamaño
            const score = this.calculateSpaceScore(space);
            this.availableSpaces.push({ ...space, score });
          }
        }
      }
    }

    // Ordenar espacios por puntuación
    this.availableSpaces.sort((a, b) => b.score - a.score);
  }

  private expandSpace(startX: number, startY: number, visited: boolean[][]): AvailableSpace | null {
    let width = 0;
    let height = 0;
    let x = startX;
    let y = startY;

    // Expandir horizontalmente
    while (x < this.spaceGrid[0].length && this.spaceGrid[y][x] && !visited[y][x]) {
      width++;
      x++;
    }

    // Expandir verticalmente
    x = startX;
    while (y < this.spaceGrid.length) {
      let canExpand = true;
      for (let i = 0; i < width; i++) {
        if (!this.spaceGrid[y][x + i] || visited[y][x + i]) {
          canExpand = false;
          break;
        }
      }
      if (!canExpand) break;
      height++;
      y++;
    }

    // Marcar como visitado
    for (let dy = 0; dy < height; dy++) {
      for (let dx = 0; dx < width; dx++) {
        visited[startY + dy][startX + dx] = true;
      }
    }

    // Convertir a coordenadas reales
    const realSpace = {
      position: {
        x: startX * this.gridSize,
        y: startY * this.gridSize
      },
      dimensions: {
        width: width * this.gridSize,
        height: height * this.gridSize
      }
    };

    // Verificar tamaño mínimo
    if (realSpace.dimensions.width >= 200 && realSpace.dimensions.height >= 150) {
      return realSpace as AvailableSpace;
    }

    return null;
  }

  private calculateSpaceScore(space: AvailableSpace): number {
    const area = space.dimensions.width * space.dimensions.height;
    const centerX = this.area.width / 2;
    const centerY = this.area.height / 2;
    const distanceToCenter = Math.sqrt(
      Math.pow(space.position.x + space.dimensions.width/2 - centerX, 2) +
      Math.pow(space.position.y + space.dimensions.height/2 - centerY, 2)
    );
    
    // Puntuación basada en área y proximidad al centro
    return area * (1 / (1 + distanceToCenter/1000));
  }

  private validateArea(area: Area): void {
    if (area.width <= 0) throw new Error('Invalid area width');
    if (area.height <= 0) throw new Error('Invalid area height');
  }

  public addChart(id: string, bounds: ChartBounds): void {
    this.validateBounds(bounds);
    this.charts.set(id, this.snapToGrid(bounds));
    this.updateSpaceGrid();
  }

  public removeChart(id: string): void {
    this.charts.delete(id);
    this.updateSpaceGrid();
  }

  public moveChart(id: string, newPosition: Position): SpaceResult {
    const chart = this.charts.get(id);
    if (!chart) throw new Error('Chart not found');

    const newBounds = {
      position: this.snapToGrid({ x: newPosition.x, y: newPosition.y }),
      dimensions: chart.dimensions
    };

    // Verificar colisiones
    const conflicts = this.findConflicts(id, newBounds);
    const available = conflicts.length === 0;

    if (available) {
      this.charts.set(id, newBounds);
      this.updateSpaceGrid();
    }

    return {
      available,
      position: newBounds.position,
      conflicts,
      availableSpaces: this.availableSpaces
    };
  }

  private findConflicts(excludeId: string, bounds: ChartBounds): ChartBounds[] {
    const conflicts: ChartBounds[] = [];
    
    this.charts.forEach((chart, id) => {
      if (id !== excludeId) {
        if (this.checkCollision(bounds, chart)) {
          conflicts.push(chart);
        }
      }
    });

    return conflicts;
  }

  private checkCollision(a: ChartBounds, b: ChartBounds): boolean {
    return (
      a.position.x < b.position.x + b.dimensions.width &&
      a.position.x + a.dimensions.width > b.position.x &&
      a.position.y < b.position.y + b.dimensions.height &&
      a.position.y + a.dimensions.height > b.position.y
    );
  }

  public resizeChart(id: string, newDimensions: Dimensions): SpaceResult {
    const chart = this.charts.get(id);
    if (!chart) throw new Error('Chart not found');

    const newBounds = {
      position: chart.position,
      dimensions: this.snapToGrid(newDimensions)
    };

    // Verificar colisiones
    const conflicts = this.findConflicts(id, newBounds);
    const available = conflicts.length === 0;

    if (available) {
      this.charts.set(id, newBounds);
      this.updateSpaceGrid();
    }

    return {
      available,
      position: newBounds.position,
      conflicts,
      availableSpaces: this.availableSpaces
    };
  }

  public findAvailableSpace(dimensions: Dimensions): Position | null {
    const snappedDimensions = this.snapToGrid(dimensions);
    
    // Buscar el mejor espacio disponible
    for (const space of this.availableSpaces) {
      if (space.dimensions.width >= snappedDimensions.width &&
          space.dimensions.height >= snappedDimensions.height) {
        return space.position;
      }
    }

    return null;
  }

  public getOptimalDistribution(): Map<string, ChartBounds> {
    // Implementar algoritmo de distribución óptima
    return new Map(this.charts);
  }

  private snapToGrid(value: Position | Dimensions): Position | Dimensions {
    if ('x' in value && 'y' in value) {
      return {
        x: Math.round(value.x / this.gridSize) * this.gridSize,
        y: Math.round(value.y / this.gridSize) * this.gridSize
      };
    } else {
      return {
        width: Math.round(value.width / this.gridSize) * this.gridSize,
        height: Math.round(value.height / this.gridSize) * this.gridSize
      };
    }
  }

  private validateBounds(bounds: ChartBounds): void {
    if (bounds.dimensions.width <= 0 || bounds.dimensions.height <= 0) {
      throw new Error('Invalid dimensions');
    }
  }

  public getChartPositions(): Map<string, ChartBounds> {
    return new Map(this.charts);
  }

  public getAvailableSpaces(): AvailableSpace[] {
    return [...this.availableSpaces];
  }

  public updateArea(newArea: Area): void {
    this.validateArea(newArea);
    this.area = newArea;
    this.initializeSpaceGrid();
    this.updateSpaceGrid();
  }
}