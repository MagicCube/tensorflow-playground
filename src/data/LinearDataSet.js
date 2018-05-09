import * as tf from '@tensorflow/tfjs';

export function generateDataSet(count = 1000, trainingRate = 0.7) {
  if (trainingRate >= 1) {
    throw new Error('Training rate must be less than 1.');
  }

  const x = tf.randomUniform([count], 0, 1);
  const y = tf(x * 0.8 + 0.2);
  const features = x.dataSync();
  const labels = y.dataSync();

  const trainingCount = Math.round(count * trainingRate);

  const training = new DataSet();
  for (let i = 0; i < trainingCount; i += 1) {
    training.add(features[i], labels[i]);
  }
  const test = new DataSet();
  for (let i = trainingCount; i < count; i += 1) {
    test.add(features[i], labels[i]);
  }

  return {
    training,
    test
  };
}

export class DataSet {
  features = [];
  labels = [];

  add(feature, label) {
    this.features.push(feature);
    this.labels.push(label);
  }

  nextBatch(batchSize = 128) {
    const indices = new Set();
    while (indices.size < batchSize) {
      indices.add(Math.ceil(Math.random() * batchSize));
    }
    const samples = {
      features: [],
      labels: []
    };
    indices.forEach(index => {
      samples.features.push(this.features[index]);
      samples.labels.push(this.labels[index]);
    });
    return samples;
  }
}
