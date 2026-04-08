const request = require('supertest');
const app = require('../../src/server');
const path = require('path');

let server;

beforeAll(() => {
  server = app.listen(0);
});

afterAll(() => {
  server.close();
});

describe('Model Loading E2E', () => {
  it('should load the default classification model', async () => {
    const res = await request(server).get('/api/models');
    expect(res.status).toBe(200);
    expect(res.body.models).toEqual(
      expect.arrayContaining([expect.objectContaining({ name: 'iris-classifier', status: 'loaded' })])
    );
  });

  it('should load a model by name and return its metadata', async () => {
    const res = await request(server).post('/api/models/load').send({ name: 'sentiment-bert' });
    expect(res.status).toBe(200);
    expect(res.body.model.name).toBe('sentiment-bert');
    expect(res.body.model.framework).toBeDefined();
    expect(res.body.model.inputShape).toBeDefined();
  });

  it('should return 404 when loading a non-existent model', async () => {
    const res = await request(server).post('/api/models/load').send({ name: 'nonexistent-model' });
    expect(res.status).toBe(404);
    expect(res.body.error).toContain('not found');
  });
});

describe('Prediction E2E', () => {
  it('should return a classification prediction with confidence scores', async () => {
    const res = await request(server)
      .post('/api/predict')
      .send({
        model: 'iris-classifier',
        input: { sepalLength: 5.1, sepalWidth: 3.5, petalLength: 1.4, petalWidth: 0.2 },
      });

    expect(res.status).toBe(200);
    expect(res.body.prediction.label).toBeDefined();
    expect(res.body.prediction.confidence).toBeGreaterThan(0);
    expect(res.body.prediction.confidence).toBeLessThanOrEqual(1);
    expect(res.body.prediction.probabilities).toBeDefined();
  });

  it('should return predictions for a batch of inputs', async () => {
    const res = await request(server)
      .post('/api/predict/batch')
      .send({
        model: 'iris-classifier',
        inputs: [
          { sepalLength: 5.1, sepalWidth: 3.5, petalLength: 1.4, petalWidth: 0.2 },
          { sepalLength: 6.7, sepalWidth: 3.0, petalLength: 5.2, petalWidth: 2.3 },
        ],
      });

    expect(res.status).toBe(200);
    expect(res.body.predictions).toHaveLength(2);
    expect(res.body.predictions[0].label).not.toBe(res.body.predictions[1].label);
  });

  it('should reject prediction requests with invalid input shape', async () => {
    const res = await request(server)
      .post('/api/predict')
      .send({ model: 'iris-classifier', input: { wrong: 'shape' } });

    expect(res.status).toBe(400);
    expect(res.body.error).toContain('input');
  });
});

describe('Prediction Accuracy Smoke Test', () => {
  it('should classify known Iris setosa sample correctly', async () => {
    const res = await request(server)
      .post('/api/predict')
      .send({
        model: 'iris-classifier',
        input: { sepalLength: 4.9, sepalWidth: 3.1, petalLength: 1.5, petalWidth: 0.1 },
      });

    expect(res.body.prediction.label).toBe('setosa');
    expect(res.body.prediction.confidence).toBeGreaterThan(0.85);
  });

  it('should classify known Iris virginica sample correctly', async () => {
    const res = await request(server)
      .post('/api/predict')
      .send({
        model: 'iris-classifier',
        input: { sepalLength: 7.7, sepalWidth: 2.8, petalLength: 6.7, petalWidth: 2.0 },
      });

    expect(res.body.prediction.label).toBe('virginica');
    expect(res.body.prediction.confidence).toBeGreaterThan(0.8);
  });
});
