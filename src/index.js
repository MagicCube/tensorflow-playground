import * as tf from '@tensorflow/tfjs';
import 'babel-polyfill';

import { generateDataSet } from './data/LinearDataSet';

window.tf = tf;

const { training, test } = generateDataSet();
const { features, labels } = training.nextBatch(2);

