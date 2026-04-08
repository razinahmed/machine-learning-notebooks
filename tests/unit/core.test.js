const { ModelRegistry } = require('../../src/models/registry');
const { Preprocessor } = require('../../src/preprocessing');
const { PredictionService } = require('../../src/prediction');
const { MetricsTracker } = require('../../src/metrics');

describe('ModelRegistry', () => {
  let registry;

  beforeEach(() => {
    registry = new ModelRegistry();
  });

  it('should register and retrieve a model by name', () => {
    const model = { name: 'iris-classifier', framework: 'sklearn', predict: jest.fn() };
    registry.register(model);
    expect(registry.get('iris-classifier')).toBe(model);
  });

  it('should list all registered model names', () => {
    registry.register({ name: 'a', framework: 'tf', predict: jest.fn() });
    registry.register({ name: 'b', framework: 'pytorch', predict: jest.fn() });
    expect(registry.listNames()).toEqual(['a', 'b']);
  });

  it('should throw when a model is not found', () => {
    expect(() => registry.get('missing')).toThrow('Model "missing" not found');
  });

  it('should unload a model and free it from the registry', () => {
    registry.register({ name: 'tmp', framework: 'tf', predict: jest.fn() });
    registry.unload('tmp');
    expect(() => registry.get('tmp')).toThrow();
  });
});

describe('Preprocessor', () => {
  it('should normalize numeric features to 0-1 range', () => {
    const preprocessor = new Preprocessor({ strategy: 'minmax', min: [4, 2, 1, 0], max: [8, 5, 7, 3] });
    const result = preprocessor.transform({ sepalLength: 6, sepalWidth: 3.5, petalLength: 4, petalWidth: 1.5 });
    result.forEach((v) => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    });
  });

  it('should standardize features with z-score normalization', () => {
    const preprocessor = new Preprocessor({
      strategy: 'zscore',
      mean: [5.8, 3.0, 3.7, 1.2],
      std: [0.8, 0.4, 1.7, 0.8],
    });
    const result = preprocessor.transform({ sepalLength: 5.8, sepalWidth: 3.0, petalLength: 3.7, petalWidth: 1.2 });
    result.forEach((v) => expect(v).toBeCloseTo(0, 1));
  });

  it('should reject input with missing features', () => {
    const preprocessor = new Preprocessor({ strategy: 'minmax', min: [0], max: [1] });
    expect(() => preprocessor.transform({})).toThrow('Missing required features');
  });
});

describe('PredictionService', () => {
  const mockModel = {
    name: 'test-model',
    predict: jest.fn((input) => ({
      label: 'setosa',
      confidence: 0.95,
      probabilities: { setosa: 0.95, versicolor: 0.03, virginica: 0.02 },
    })),
  };

  it('should return a prediction with label and confidence', () => {
    const service = new PredictionService(mockModel);
    const result = service.predict([0.5, 0.3, 0.1, 0.05]);
    expect(result.label).toBe('setosa');
    expect(result.confidence).toBe(0.95);
  });

  it('should validate that input is a numeric array', () => {
    const service = new PredictionService(mockModel);
    expect(() => service.predict('not an array')).toThrow('Input must be a numeric array');
  });

  it('should call the model predict function with the input', () => {
    const service = new PredictionService(mockModel);
    service.predict([1, 2, 3, 4]);
    expect(mockModel.predict).toHaveBeenCalledWith([1, 2, 3, 4]);
  });
});

describe('MetricsTracker', () => {
  it('should compute accuracy from prediction results', () => {
    const tracker = new MetricsTracker();
    tracker.record({ predicted: 'a', actual: 'a' });
    tracker.record({ predicted: 'b', actual: 'b' });
    tracker.record({ predicted: 'a', actual: 'b' });
    expect(tracker.accuracy()).toBeCloseTo(0.667, 2);
  });

  it('should compute precision per class', () => {
    const tracker = new MetricsTracker();
    tracker.record({ predicted: 'a', actual: 'a' });
    tracker.record({ predicted: 'a', actual: 'b' });
    tracker.record({ predicted: 'b', actual: 'b' });
    expect(tracker.precision('a')).toBeCloseTo(0.5);
    expect(tracker.precision('b')).toBeCloseTo(1.0);
  });

  it('should reset metrics', () => {
    const tracker = new MetricsTracker();
    tracker.record({ predicted: 'a', actual: 'a' });
    tracker.reset();
    expect(tracker.accuracy()).toBe(0);
  });
});
